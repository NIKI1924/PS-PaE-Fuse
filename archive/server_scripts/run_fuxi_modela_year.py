#!/usr/bin/env python3
"""Generate canonical FuXi Model-A forecasts from public ARCO-ERA5 inputs.

The output contract is deliberately small and resumable:

* one atomic NPZ plus one JSON manifest per 00 UTC initialization;
* seven 24-hour leads (24..168 h), not all 28 six-hour model steps;
* six Wenqiong variables on the canonical 721x1440, north-to-south grid;
* no silent NaN/Inf replacement;
* deterministic sharding for multi-GPU year runs.

This script is intended to run in the server's FuXi environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

import gcsfs
import numpy as np
import onnxruntime as ort
import pandas as pd
import xarray as xr


ARCO = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_MODELS = Path("/tank2/weatherModels/FuXi/fuxi_models")
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
STAGES = (("short", 20), ("medium", 8))
TARGET_STEPS = (4, 8, 12, 16, 20, 24, 28)
TARGET_LEADS = tuple(step * 6 for step in TARGET_STEPS)
N_LAT, N_LON = 721, 1440
WQ_CHANNELS = (
    ("2m_temperature", 65),
    ("10m_u_component_of_wind", 66),
    ("10m_v_component_of_wind", 67),
    ("mean_sea_level_pressure", 68),
    ("geopotential_500", 7),
    ("temperature_850", 23),
)
T = TypeVar("T")


def retry_read(fn: Callable[[], T], label: str, timeout: int = 240, retries: int = 10) -> T:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        pool = ThreadPoolExecutor(1)
        future = pool.submit(fn)
        try:
            value = future.result(timeout=timeout)
            pool.shutdown(wait=False)
            return value
        except Exception as exc:
            last = exc
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            wait = min(10 * attempt, 60)
            print(
                f"[read retry {attempt}/{retries}] {label}: {type(exc).__name__}; wait {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"ARCO read failed for {label}") from last


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary) as check:
            expected = {name for name, _index in WQ_CHANNELS}
            if set(check.files) != expected:
                raise ValueError(f"NPZ variables mismatch: {sorted(check.files)}")
            for name in expected:
                if check[name].shape != (len(TARGET_LEADS), N_LAT, N_LON):
                    raise ValueError(f"{name}: bad stored shape {check[name].shape}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_humidity_from_q(q: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    pressure = np.asarray(LEVELS, np.float32)[None, :, None, None]
    vapour_pressure = q * pressure / (0.622 + 0.378 * q)
    saturation = 6.112 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
    return np.clip(100.0 * vapour_pressure / saturation, 0.0, 100.0).astype(np.float32)


def time_encoding(init_time: pd.Timestamp, total_steps: int = 28, frequency_hours: int = 6) -> np.ndarray:
    initial = np.array([init_time])
    embeddings = []
    for step in range(total_steps):
        offsets = np.array(
            [pd.Timedelta(hours=value * frequency_hours) for value in (step - 1, step, step + 1)]
        )
        times = initial[:, None] + offsets[None]
        periods = [pd.Period(value, "h") for value in times.reshape(-1)]
        cyclical = np.array(
            [(period.day_of_year / 366.0, period.hour / 24.0) for period in periods],
            dtype=np.float32,
        )
        embeddings.append(
            np.concatenate([np.sin(cyclical), np.cos(cyclical)], axis=-1).reshape(1, -1)
        )
    return np.stack(embeddings)


def array_stats(array: np.ndarray) -> dict[str, float | list[int] | str]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "finite_fraction": float(np.isfinite(array).mean()),
    }


class FuXiModelA:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.sessions: dict[str, ort.InferenceSession] = {}
        print("[open ARCO metadata]", flush=True)
        filesystem = gcsfs.GCSFileSystem(token="anon")
        self.dataset = retry_read(
            lambda: xr.open_zarr(
                filesystem.get_mapper(ARCO), consolidated=True, chunks={"time": 1}
            ),
            "open ARCO",
            timeout=300,
        )
        latitude = self.dataset.latitude.values
        longitude = self.dataset.longitude.values
        if not (
            latitude.shape == (N_LAT,)
            and np.isclose(latitude[0], 90.0)
            and np.isclose(latitude[-1], -90.0)
            and longitude.shape == (N_LON,)
            and np.isclose(longitude[0], 0.0)
            and np.isclose(longitude[-1], 359.75)
        ):
            raise ValueError("ARCO grid is not canonical 721x1440 north-to-south")
        self.has_relative_humidity = "relative_humidity" in self.dataset.data_vars

    def close(self) -> None:
        self.dataset.close()
        self.sessions.clear()

    def session(self, stage: str) -> ort.InferenceSession:
        if stage not in self.sessions:
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.enable_mem_reuse = False
            options.intra_op_num_threads = 1
            model = self.model_dir / f"{stage}.onnx"
            self.sessions[stage] = ort.InferenceSession(
                str(model),
                sess_options=options,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"arena_extend_strategy": "kSameAsRequested"},
                    )
                ],
            )
            providers = self.sessions[stage].get_providers()
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{stage}: CUDAExecutionProvider is not active: {providers}")
            print(f"[loaded {stage} on CUDA]", flush=True)
        return self.sessions[stage]

    def build_input(self, init_time: pd.Timestamp) -> tuple[np.ndarray, dict]:
        input_times = [init_time - pd.Timedelta(hours=6), init_time]
        output = np.empty((2, 70, N_LAT, N_LON), dtype=np.float32)
        started = time.time()
        z = retry_read(
            lambda: self.dataset.geopotential.sel(time=input_times, level=LEVELS).values,
            "geopotential",
        ).astype(np.float32)
        temperature = retry_read(
            lambda: self.dataset.temperature.sel(time=input_times, level=LEVELS).values,
            "temperature",
        ).astype(np.float32)
        u_wind = retry_read(
            lambda: self.dataset.u_component_of_wind.sel(time=input_times, level=LEVELS).values,
            "u wind",
        ).astype(np.float32)
        v_wind = retry_read(
            lambda: self.dataset.v_component_of_wind.sel(time=input_times, level=LEVELS).values,
            "v wind",
        ).astype(np.float32)
        if self.has_relative_humidity:
            relative_humidity = retry_read(
                lambda: self.dataset.relative_humidity.sel(time=input_times, level=LEVELS).values,
                "relative humidity",
            ).astype(np.float32)
            humidity_source = "ARCO relative_humidity"
        else:
            specific_humidity = retry_read(
                lambda: self.dataset.specific_humidity.sel(time=input_times, level=LEVELS).values,
                "specific humidity",
            ).astype(np.float32)
            relative_humidity = relative_humidity_from_q(specific_humidity, temperature)
            humidity_source = "Bolton conversion from ARCO specific_humidity"

        for channel, variable in (
            (65, "2m_temperature"),
            (66, "10m_u_component_of_wind"),
            (67, "10m_v_component_of_wind"),
            (68, "mean_sea_level_pressure"),
        ):
            output[:, channel] = retry_read(
                lambda variable=variable: self.dataset[variable].sel(time=input_times).values,
                variable,
            ).astype(np.float32)

        precipitation = np.zeros((2, N_LAT, N_LON), dtype=np.float32)
        for index, timestamp in enumerate(input_times):
            hours = pd.date_range(timestamp - pd.Timedelta(hours=5), timestamp, freq="1h")
            precipitation[index] = retry_read(
                lambda hours=hours: self.dataset.total_precipitation.sel(time=list(hours))
                .sum("time")
                .values,
                f"total precipitation frame {index}",
            ).astype(np.float32)

        output[:, 0:13] = z
        output[:, 13:26] = temperature
        output[:, 26:39] = u_wind
        output[:, 39:52] = v_wind
        output[:, 52:65] = relative_humidity
        output[:, 69] = precipitation
        if not np.isfinite(output).all():
            raise ValueError(
                f"FuXi input contains NaN/Inf: finite_fraction={np.isfinite(output).mean():.8f}"
            )
        details = {
            "input_times": [timestamp.isoformat() for timestamp in input_times],
            "humidity_source": humidity_source,
            "build_seconds": round(time.time() - started, 3),
            "summary": array_stats(output),
        }
        return output, details

    def run(self, init_time: pd.Timestamp) -> tuple[dict[str, np.ndarray], dict]:
        started = time.time()
        input_frames, input_manifest = self.build_input(init_time)
        state = input_frames[None]
        embeddings = time_encoding(init_time)
        selected = {name: [] for name, _index in WQ_CHANNELS}
        global_step = 0
        for stage, count in STAGES:
            session = self.session(stage)
            for _local_step in range(count):
                state, = session.run(None, {"input": state, "temb": embeddings[global_step]})
                global_step += 1
                if global_step in TARGET_STEPS:
                    frame = np.asarray(state[0, -1])
                    for name, channel in WQ_CHANNELS:
                        # The model input and output are already N->S. Do not flip here.
                        selected[name].append(frame[channel].astype(np.float32, copy=True))
                if global_step in TARGET_STEPS or global_step == 1:
                    print(
                        f"  step={global_step}/28 lead={global_step * 6}h "
                        f"selected={global_step in TARGET_STEPS}",
                        flush=True,
                    )
        output = {name: np.stack(frames) for name, frames in selected.items()}
        for name, array in output.items():
            if array.shape != (len(TARGET_LEADS), N_LAT, N_LON):
                raise ValueError(f"{name}: unexpected output shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name}: NaN/Inf output; finite_fraction={np.isfinite(array).mean():.8f}"
                )
        manifest = {
            "status": "complete",
            "spec": "wenqiong-modela-fuxi-v1",
            "init_time": init_time.isoformat(),
            "lead_hours": list(TARGET_LEADS),
            "latitude_orientation": "north_to_south",
            "longitude_convention": "0_to_359.75",
            "grid_shape": [N_LAT, N_LON],
            "pressure_levels_top_to_bottom_hpa": LEVELS,
            "arco_source": ARCO,
            "stages": [{"name": name, "steps": count} for name, count in STAGES],
            "input": input_manifest,
            "outputs": {name: array_stats(array) for name, array in output.items()},
            "elapsed_seconds": round(time.time() - started, 3),
        }
        return output, manifest


def valid_existing(npz_path: Path, manifest_path: Path) -> bool:
    if not npz_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("lead_hours") != list(TARGET_LEADS):
            return False
        with np.load(npz_path) as archive:
            if set(archive.files) != {name for name, _index in WQ_CHANNELS}:
                return False
            return all(
                archive[name].shape == (len(TARGET_LEADS), N_LAT, N_LON)
                for name, _index in WQ_CHANNELS
            )
    except Exception:
        return False


def run_one(runner: FuXiModelA, timestamp: pd.Timestamp, output_dir: Path) -> str:
    if timestamp.hour != 0 or timestamp.minute or timestamp.second:
        raise ValueError("Model-A initialization must be exactly 00 UTC")
    tag = timestamp.strftime("%Y%m%d%H")
    npz_path = output_dir / f"fuxi_{tag}.npz"
    manifest_path = output_dir / f"fuxi_{tag}.json"
    if valid_existing(npz_path, manifest_path):
        print(f"[{tag}] complete; skip", flush=True)
        return "skipped"
    print(f"[{tag}] start", flush=True)
    arrays, manifest = runner.run(timestamp)
    atomic_npz(npz_path, arrays)
    manifest["npz_file"] = npz_path.name
    manifest["npz_bytes"] = npz_path.stat().st_size
    atomic_json(manifest_path, manifest)
    print(
        f"[{tag}] committed {npz_path.name} {npz_path.stat().st_size / 2**30:.3f} GiB "
        f"elapsed={manifest['elapsed_seconds']}s",
        flush=True,
    )
    return "created"


def build_dates(args: argparse.Namespace) -> pd.DatetimeIndex:
    if args.date:
        dates = pd.DatetimeIndex([pd.Timestamp(args.date)])
    else:
        if not args.start or not args.end:
            raise ValueError("provide --date or both --start and --end")
        dates = pd.date_range(args.start, args.end, freq="1D")
    dates = dates[(np.arange(len(dates)) % args.shard_count) == args.shard_index]
    return dates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.shard_count <= 16:
        raise ValueError("--shard-count must be in [1,16]")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid --shard-index")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    dates = build_dates(args)
    run_config = {
        "spec": "wenqiong-modela-fuxi-v1",
        "dates": {
            "count_this_shard": int(len(dates)),
            "first": dates[0].isoformat() if len(dates) else None,
            "last": dates[-1].isoformat() if len(dates) else None,
        },
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "lead_hours": list(TARGET_LEADS),
        "latitude_orientation": "north_to_south",
        "arco_source": ARCO,
        "model_files": {
            stage: {
                "path": str(args.model_dir / f"{stage}.onnx"),
                "sha256": sha256(args.model_dir / f"{stage}.onnx"),
            }
            for stage, _count in STAGES
        },
        "script_sha256": sha256(Path(__file__)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_json(args.output / f"_run_config_shard{args.shard_index}.json", run_config)

    runner = FuXiModelA(args.model_dir)
    failures: dict[str, str] = {}
    created = skipped = 0
    try:
        for timestamp in dates:
            for attempt in range(1, args.max_attempts + 1):
                try:
                    status = run_one(runner, timestamp, args.output)
                    created += status == "created"
                    skipped += status == "skipped"
                    break
                except Exception as exc:
                    print(
                        f"[{timestamp.isoformat()}] attempt {attempt}/{args.max_attempts} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if attempt == args.max_attempts:
                        failures[timestamp.isoformat()] = f"{type(exc).__name__}: {exc}"
                    else:
                        time.sleep(30 * attempt)
    finally:
        runner.close()

    summary = {
        "status": "complete" if not failures else "partial",
        "created": created,
        "skipped": skipped,
        "failed": failures,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    atomic_json(args.output / f"_summary_shard{args.shard_index}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
