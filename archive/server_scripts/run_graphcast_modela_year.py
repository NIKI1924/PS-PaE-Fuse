#!/usr/bin/env python3
"""Generate a fixed-checkpoint GraphCast member for Wenqiong Model A.

The script intentionally uses one process per GPU and one contiguous date
range per process.  It reads the two ERA5 initialization frames from ARCO,
runs the official 37-level GraphCast research checkpoint to 168 hours, and
stores only Wenqiong's six variables at seven daily lead times.

Outputs are committed atomically as one NPZ plus one JSON manifest per init.
The canonical output grid is 721x1440 with latitude north-to-south.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

# Avoid JAX reserving the complete GPU and leave headroom for the host display,
# drivers, and transient allocations.  CUDA_VISIBLE_DEVICES is set by the
# launcher before this process starts.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def first_existing(candidates: list[str], label: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"No {label} found in: {candidates}")


WORKDIR = Path(
    os.environ.get(
        "GC_WORKDIR",
        str(first_existing(["/home/xrx/gc_2021", "/home/deploy/gc"], "GraphCast workdir")),
    )
)
os.chdir(WORKDIR)

import dask  # noqa: E402
import gcsfs  # noqa: E402
import haiku as hk  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from graphcast import (  # noqa: E402
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)


SPEC = "wenqiong-modela-graphcast-fixed-1979-2017-v1"
ARCO = "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CHECKPOINT_NAME = (
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - "
    "mesh 2to6 - precipitation input and output.npz"
)
PARAMS = Path(
    os.environ.get(
        "GC_PARAMS",
        str(
            first_existing(
                [
                    f"/tank2/weatherModels/GraphCast/params/{CHECKPOINT_NAME}",
                    f"/home/deploy/GraphCast/params/{CHECKPOINT_NAME}",
                ],
                "GraphCast checkpoint",
            )
        ),
    )
)
STATS = Path(
    os.environ.get(
        "GC_STATS",
        str(
            first_existing(
                ["/tank2/weatherModels/GraphCast/stats", "/home/deploy/GraphCast/stats"],
                "GraphCast statistics",
            )
        ),
    )
)
STATICS = Path(
    os.environ.get(
        "GC_STATICS",
        str(first_existing(["/home/deploy/gc/statics.npz"], "GraphCast statics")),
    )
)
CACHE_DIR = Path(
    os.environ.get(
        "GC_JAX_CACHE",
        "/tank2/xrx/jax_cache" if Path("/tank2/xrx").exists() else "/mnt/a/weather-ZJJ/jax_cache",
    )
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
N_LAT, N_LON = 721, 1440
N_STEPS = 28
TARGET_STEPS = (4, 8, 12, 16, 20, 24, 28)
TARGET_INDICES = np.asarray([step - 1 for step in TARGET_STEPS], dtype=np.int64)
TARGET_LEADS = tuple(step * 6 for step in TARGET_STEPS)
ATMOSPHERIC = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
SURFACE = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
WQ_OUTPUTS = (
    ("2m_temperature", None, "2m_temperature"),
    ("10m_u_component_of_wind", None, "10m_u_component_of_wind"),
    ("10m_v_component_of_wind", None, "10m_v_component_of_wind"),
    ("mean_sea_level_pressure", None, "mean_sea_level_pressure"),
    ("geopotential", 500, "geopotential_500"),
    ("temperature", 850, "temperature_850"),
)
OUTPUT_NAMES = {name for _variable, _level, name in WQ_OUTPUTS}
T = TypeVar("T")


def retry_read(
    function: Callable[[], T],
    label: str,
    timeout: int = 300,
    retries: int = 6,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        pool = ThreadPoolExecutor(1)
        future = pool.submit(function)
        try:
            value = future.result(timeout=timeout)
            pool.shutdown(wait=False)
            return value
        except Exception as error:
            last_error = error
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            wait_seconds = min(10 * attempt, 60)
            print(
                f"[read retry {attempt}/{retries}] {label}: {type(error).__name__}; "
                f"wait {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"ARCO read failed for {label}") from last_error


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
        with np.load(temporary) as archive:
            if set(archive.files) != OUTPUT_NAMES:
                raise ValueError(f"stored variables differ: {sorted(archive.files)}")
            for name in OUTPUT_NAMES:
                if archive[name].shape != (len(TARGET_LEADS), N_LAT, N_LON):
                    raise ValueError(f"{name}: bad stored shape {archive[name].shape}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def array_stats(array: np.ndarray) -> dict[str, float | list[int]]:
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean(dtype=np.float64)),
    }


class GraphCastModelA:
    def __init__(self) -> None:
        print(f"[load checkpoint] {PARAMS}", flush=True)
        with PARAMS.open("rb") as stream:
            loaded = checkpoint.load(stream, graphcast.CheckPoint)
        self.params = loaded.params
        self.model_config = loaded.model_config
        self.task_config = loaded.task_config
        self.state = {}
        self.stats = {
            name: xr.load_dataset(STATS / f"{name}.nc").compute()
            for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")
        }

        print("[open ARCO ERA5 metadata]", flush=True)
        filesystem = gcsfs.GCSFileSystem(token="anon")
        self.dataset = retry_read(
            lambda: xr.open_zarr(
                filesystem.get_mapper(ARCO),
                consolidated=True,
                chunks={"time": 1},
            ),
            "open ARCO",
            timeout=300,
        )
        available_levels = [int(value) for value in self.dataset.level.values]
        missing_levels = sorted(set(LEVELS) - set(available_levels))
        if missing_levels:
            raise ValueError(f"ARCO is missing GraphCast levels: {missing_levels}")
        required_variables = set(ATMOSPHERIC + SURFACE) | {
            "toa_incident_solar_radiation",
            "total_precipitation",
        }
        missing_variables = sorted(required_variables - set(self.dataset.data_vars))
        if missing_variables:
            raise ValueError(f"ARCO is missing variables: {missing_variables}")

        def construct_predictor(model_config, task_config):
            predictor = graphcast.GraphCast(model_config, task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.stats["diffs_stddev_by_level"],
                mean_by_level=self.stats["mean_by_level"],
                stddev_by_level=self.stats["stddev_by_level"],
            )
            return autoregressive.Predictor(predictor, gradient_checkpointing=True)

        @hk.transform_with_state
        def forward(model_config, task_config, inputs, targets_template, forcings):
            return construct_predictor(model_config, task_config)(
                inputs,
                targets_template=targets_template,
                forcings=forcings,
            )

        compiled = jax.jit(
            functools.partial(
                forward.apply,
                self.params,
                self.state,
                jax.random.PRNGKey(0),
                self.model_config,
                self.task_config,
            )
        )

        def run_forward(_rng, inputs, targets_template, forcings):
            return compiled(inputs, targets_template, forcings)[0]

        self.run_forward = run_forward

    def close(self) -> None:
        self.dataset.close()
        for dataset in self.stats.values():
            dataset.close()

    def build_batch(self, init_time: pd.Timestamp) -> tuple[xr.Dataset, dict]:
        started = time.time()
        all_times = pd.date_range(
            init_time - pd.Timedelta(hours=6),
            periods=2 + N_STEPS,
            freq="6h",
        )
        input_times = list(all_times[:2])
        latitude = self.dataset.latitude.values.astype(np.float32)
        longitude = self.dataset.longitude.values.astype(np.float32)
        flip_to_south_north = bool(latitude[0] > latitude[-1])
        if flip_to_south_north:
            latitude = latitude[::-1]

        def orient(array: np.ndarray) -> np.ndarray:
            values = np.asarray(array)
            if flip_to_south_north:
                values = values[..., ::-1, :]
            return values.copy()

        height, width, n_level, n_time = N_LAT, N_LON, len(LEVELS), 2 + N_STEPS
        fields: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

        static_archive = np.load(STATICS)
        surface_geopotential = static_archive["geopotential_at_surface"]
        land_sea_mask = static_archive["land_sea_mask"]
        static_latitude = static_archive["lat"]
        if bool(static_latitude[0] < static_latitude[-1]) != bool(latitude[0] < latitude[-1]):
            surface_geopotential = surface_geopotential[::-1]
            land_sea_mask = land_sea_mask[::-1]
        fields["geopotential_at_surface"] = (
            ("lat", "lon"),
            surface_geopotential.astype(np.float32),
        )
        fields["land_sea_mask"] = (("lat", "lon"), land_sea_mask.astype(np.float32))

        def load_surface():
            with dask.config.set(scheduler="single-threaded"):
                return self.dataset[list(SURFACE)].sel(time=input_times).compute()

        surface = retry_read(load_surface, "surface")
        for variable in SURFACE:
            array = np.full((1, n_time, height, width), np.nan, np.float32)
            array[0, :2] = orient(surface[variable].values)
            fields[variable] = (("batch", "time", "lat", "lon"), array)
        surface.close()

        def load_toa():
            with dask.config.set(scheduler="single-threaded"):
                return self.dataset["toa_incident_solar_radiation"].sel(time=list(all_times)).values

        toa = retry_read(load_toa, "toa forcing", timeout=600).astype(np.float32)
        array = np.full((1, n_time, height, width), np.nan, np.float32)
        array[0] = orient(toa)
        fields["toa_incident_solar_radiation"] = (
            ("batch", "time", "lat", "lon"),
            array,
        )
        del toa

        precipitation_hours = pd.date_range(
            input_times[0] - pd.Timedelta(hours=5),
            input_times[1],
            freq="1h",
        )

        def load_precipitation():
            with dask.config.set(scheduler="single-threaded"):
                return self.dataset["total_precipitation"].sel(
                    time=list(precipitation_hours)
                ).compute()

        precipitation = retry_read(load_precipitation, "precipitation")
        array = np.full((1, n_time, height, width), np.nan, np.float32)
        for index, timestamp in enumerate(input_times):
            window = pd.date_range(timestamp - pd.Timedelta(hours=5), timestamp, freq="1h")
            array[0, index] = orient(
                precipitation.sel(time=list(window)).sum("time").values.astype(np.float32)
            )
        fields["total_precipitation_6hr"] = (
            ("batch", "time", "lat", "lon"),
            array,
        )
        precipitation.close()

        for variable in ATMOSPHERIC:
            def load_atmosphere(variable_name=variable):
                with dask.config.set(scheduler="single-threaded"):
                    return self.dataset[variable_name].sel(
                        time=input_times,
                        level=LEVELS,
                    ).values

            values = retry_read(load_atmosphere, f"atmosphere:{variable}").astype(np.float32)
            array = np.full((1, n_time, n_level, height, width), np.nan, np.float32)
            array[0, :2] = orient(values)
            fields[variable] = (("batch", "time", "level", "lat", "lon"), array)
            del values

        relative_time = (
            np.arange(n_time) * 6 * 3600 * 1_000_000_000
        ).astype("timedelta64[ns]")
        batch = xr.Dataset(
            fields,
            coords={
                "lon": ("lon", longitude),
                "lat": ("lat", latitude),
                "level": ("level", np.asarray(LEVELS, np.int32)),
                "time": ("time", relative_time),
                "datetime": (("batch", "time"), np.asarray(all_times).reshape(1, n_time)),
            },
        )
        details = {
            "input_times": [timestamp.isoformat() for timestamp in input_times],
            "source_latitude_orientation": "north_to_south"
            if flip_to_south_north
            else "south_to_north",
            "graphcast_input_orientation": "south_to_north",
            "read_scheduler": "single-threaded",
            "build_seconds": round(time.time() - started, 3),
        }
        return batch, details

    def run(self, init_time: pd.Timestamp) -> tuple[dict[str, np.ndarray], dict]:
        started = time.time()
        example_batch, input_details = self.build_batch(init_time)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{N_STEPS * 6}h"),
            **dataclasses.asdict(self.task_config),
        )
        accumulated: dict[str, list[np.ndarray]] = {name: [] for name in OUTPUT_NAMES}
        for chunk in rollout.chunked_prediction_generator(
            self.run_forward,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
            num_steps_per_chunk=2,
        ):
            for variable, level, output_name in WQ_OUTPUTS:
                data_array = chunk[variable].isel(batch=0)
                if level is not None:
                    data_array = data_array.sel(level=level)
                accumulated[output_name].append(
                    data_array.transpose("time", "lat", "lon").values.astype(np.float32)
                )
            del chunk

        arrays: dict[str, np.ndarray] = {}
        for output_name in OUTPUT_NAMES:
            all_steps = np.concatenate(accumulated[output_name], axis=0)
            if all_steps.shape != (N_STEPS, N_LAT, N_LON):
                raise ValueError(f"{output_name}: bad rollout shape {all_steps.shape}")
            # GraphCast predicts south-to-north; Model A stores north-to-south.
            selected = all_steps[TARGET_INDICES, ::-1, :].copy()
            if not np.isfinite(selected).all():
                raise ValueError(
                    f"{output_name}: non-finite fraction "
                    f"{1.0 - float(np.isfinite(selected).mean()):.8f}"
                )
            arrays[output_name] = selected
            del all_steps

        manifest = {
            "status": "complete",
            "spec": SPEC,
            "init_time": init_time.isoformat(),
            "lead_hours": list(TARGET_LEADS),
            "variables": sorted(OUTPUT_NAMES),
            "grid": {
                "latitude": "90_to_-90",
                "longitude": "0_to_359.75",
                "shape": [N_LAT, N_LON],
            },
            "checkpoint": {
                "path": str(PARAMS),
                "sha256": sha256(PARAMS),
                "training_period": "ERA5 1979-2017",
            },
            "era5_source": ARCO,
            "input": input_details,
            "output_summary": {name: array_stats(array) for name, array in arrays.items()},
            "elapsed_seconds": round(time.time() - started, 3),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        del example_batch, inputs, targets, forcings
        gc.collect()
        return arrays, manifest


def valid_existing(npz_path: Path, manifest_path: Path) -> bool:
    if not npz_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("spec") != SPEC
            or manifest.get("lead_hours") != list(TARGET_LEADS)
        ):
            return False
        with np.load(npz_path) as archive:
            return set(archive.files) == OUTPUT_NAMES and all(
                archive[name].shape == (len(TARGET_LEADS), N_LAT, N_LON)
                for name in OUTPUT_NAMES
            )
    except Exception:
        return False


def run_one(runner: GraphCastModelA, timestamp: pd.Timestamp, output_dir: Path) -> str:
    tag = timestamp.strftime("%Y%m%d%H")
    npz_path = output_dir / f"graphcast_{tag}.npz"
    manifest_path = output_dir / f"graphcast_{tag}.json"
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
    elif args.start and args.end:
        dates = pd.date_range(args.start, args.end, freq="1D")
    else:
        raise ValueError("provide --date or both --start and --end")
    if any(timestamp.hour != 0 for timestamp in dates):
        raise ValueError("Model A initialization times must be 00 UTC")
    return dates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    dates = build_dates(args)
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint_sha256 = sha256(PARAMS)
    config = {
        "spec": SPEC,
        "worker_id": args.worker_id,
        "dates": {
            "count": len(dates),
            "first": dates[0].isoformat(),
            "last": dates[-1].isoformat(),
        },
        "lead_hours": list(TARGET_LEADS),
        "variables": sorted(OUTPUT_NAMES),
        "latitude_orientation": "north_to_south",
        "checkpoint": {"path": str(PARAMS), "sha256": checkpoint_sha256},
        "era5_source": ARCO,
        "script_sha256": sha256(Path(__file__)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host": os.uname().nodename,
    }
    atomic_json(args.output / f"_run_config_{args.worker_id}.json", config)

    runner = GraphCastModelA()
    created = skipped = 0
    failures: dict[str, str] = {}
    try:
        for timestamp in dates:
            for attempt in range(1, args.max_attempts + 1):
                try:
                    status = run_one(runner, timestamp, args.output)
                    created += int(status == "created")
                    skipped += int(status == "skipped")
                    break
                except Exception as error:
                    print(
                        f"[{timestamp.isoformat()}] attempt {attempt}/{args.max_attempts} failed: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    gc.collect()
                    if attempt == args.max_attempts:
                        failures[timestamp.isoformat()] = f"{type(error).__name__}: {error}"
                    else:
                        time.sleep(30 * attempt)
    finally:
        runner.close()

    summary = {
        "status": "complete" if not failures else "partial",
        "worker_id": args.worker_id,
        "created": created,
        "skipped": skipped,
        "failed": failures,
    }
    atomic_json(args.output / f"_summary_{args.worker_id}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
