#!/usr/bin/env python3
"""FuXi Model-A v2: use WB2 ERA5's native 13-level FuXi input fields.

This module reuses the inference/output machinery from v1, but replaces the
ARCO q-to-RH approximation with WB2 ERA5 relative humidity. WB2 stores RH as a
0..1 fraction; the official FuXi input uses percent, so v2 validates the range
and multiplies by 100 explicitly. Reads are grouped and forced through Dask's
single-threaded scheduler to avoid stressing the shared network gateway.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import dask
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

import run_fuxi_modela_year as base


ERA5_SOURCE = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
SPEC = "wenqiong-modela-fuxi-v2-wb2era5"
PRESSURE_VARIABLES = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "relative_humidity",
)
SURFACE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
)


class FuXiModelAWB2(base.FuXiModelA):
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.sessions = {}
        print("[open WB2 ERA5 metadata]", flush=True)
        filesystem = gcsfs.GCSFileSystem(token="anon")
        self.dataset = base.retry_read(
            lambda: xr.open_zarr(
                filesystem.get_mapper(ERA5_SOURCE),
                consolidated=True,
                chunks={"time": 1},
            ),
            "open WB2 ERA5",
            timeout=300,
        )
        latitude = self.dataset.latitude.values
        longitude = self.dataset.longitude.values
        if not (
            latitude.shape == (base.N_LAT,)
            and np.isclose(latitude[0], 90.0)
            and np.isclose(latitude[-1], -90.0)
            and np.all(np.diff(latitude) < 0)
            and longitude.shape == (base.N_LON,)
            and np.isclose(longitude[0], 0.0)
            and np.isclose(longitude[-1], 359.75)
            and np.all(np.diff(longitude) > 0)
        ):
            raise ValueError("WB2 ERA5 grid is not canonical 721x1440 N->S")
        available_levels = [int(value) for value in self.dataset.level.values]
        if available_levels != base.LEVELS:
            raise ValueError(
                f"WB2 ERA5 levels differ from FuXi order: {available_levels}"
            )
        required = set(PRESSURE_VARIABLES + SURFACE_VARIABLES)
        missing = sorted(required - set(self.dataset.data_vars))
        if missing:
            raise KeyError(f"WB2 ERA5 missing FuXi inputs: {missing}")

    def _load_group(self, variables: tuple[str, ...], times: list[pd.Timestamp], levels=False):
        def fetch():
            selection = self.dataset[list(variables)].sel(time=times)
            if levels:
                selection = selection.sel(level=base.LEVELS)
            with dask.config.set(scheduler="single-threaded"):
                return selection.compute()

        return base.retry_read(fetch, "+".join(variables), timeout=900, retries=6)

    def build_input(self, init_time: pd.Timestamp) -> tuple[np.ndarray, dict]:
        input_times = [init_time - pd.Timedelta(hours=6), init_time]
        started = time.time()
        pressure = self._load_group(PRESSURE_VARIABLES, input_times, levels=True)
        surface = self._load_group(SURFACE_VARIABLES, input_times, levels=False)

        rh_fraction = pressure.relative_humidity.values.astype(np.float32)
        rh_min = float(rh_fraction.min())
        rh_max = float(rh_fraction.max())
        if rh_min < -0.05 or rh_max > 1.5:
            raise ValueError(
                f"WB2 relative_humidity is not a fraction: min={rh_min}, max={rh_max}"
            )
        relative_humidity_percent = np.clip(rh_fraction * 100.0, 0.0, 100.0)

        output = np.empty((2, 70, base.N_LAT, base.N_LON), dtype=np.float32)
        output[:, 0:13] = pressure.geopotential.values.astype(np.float32)
        output[:, 13:26] = pressure.temperature.values.astype(np.float32)
        output[:, 26:39] = pressure.u_component_of_wind.values.astype(np.float32)
        output[:, 39:52] = pressure.v_component_of_wind.values.astype(np.float32)
        output[:, 52:65] = relative_humidity_percent
        output[:, 65] = surface["2m_temperature"].values.astype(np.float32)
        output[:, 66] = surface["10m_u_component_of_wind"].values.astype(np.float32)
        output[:, 67] = surface["10m_v_component_of_wind"].values.astype(np.float32)
        output[:, 68] = surface["mean_sea_level_pressure"].values.astype(np.float32)
        output[:, 69] = surface["total_precipitation_6hr"].values.astype(np.float32)
        pressure.close()
        surface.close()

        if not np.isfinite(output).all():
            raise ValueError(
                f"FuXi input has NaN/Inf: finite_fraction={np.isfinite(output).mean():.8f}"
            )
        details = {
            "input_times": [timestamp.isoformat() for timestamp in input_times],
            "humidity_source": "WB2 ERA5 RH fraction multiplied by 100",
            "humidity_fraction_range_before_conversion": [rh_min, rh_max],
            "precipitation_source": "WB2 ERA5 total_precipitation_6hr",
            "read_scheduler": "single-threaded",
            "build_seconds": round(time.time() - started, 3),
            "summary": base.array_stats(output),
        }
        return output, details

    def run(self, init_time: pd.Timestamp):
        arrays, manifest = super().run(init_time)
        manifest["spec"] = SPEC
        manifest.pop("arco_source", None)
        manifest["era5_source"] = ERA5_SOURCE
        return arrays, manifest


def valid_existing(npz_path: Path, manifest_path: Path) -> bool:
    if not npz_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("spec") != SPEC
            or manifest.get("lead_hours") != list(base.TARGET_LEADS)
        ):
            return False
        with np.load(npz_path) as archive:
            return set(archive.files) == {name for name, _channel in base.WQ_CHANNELS} and all(
                archive[name].shape
                == (len(base.TARGET_LEADS), base.N_LAT, base.N_LON)
                for name, _channel in base.WQ_CHANNELS
            )
    except Exception:
        return False


def run_one(runner: FuXiModelAWB2, timestamp: pd.Timestamp, output_dir: Path) -> str:
    tag = timestamp.strftime("%Y%m%d%H")
    npz_path = output_dir / f"fuxi_{tag}.npz"
    manifest_path = output_dir / f"fuxi_{tag}.json"
    if valid_existing(npz_path, manifest_path):
        print(f"[{tag}] complete; skip", flush=True)
        return "skipped"
    print(f"[{tag}] start", flush=True)
    arrays, manifest = runner.run(timestamp)
    base.atomic_npz(npz_path, arrays)
    manifest["npz_file"] = npz_path.name
    manifest["npz_bytes"] = npz_path.stat().st_size
    base.atomic_json(manifest_path, manifest)
    print(
        f"[{tag}] committed {npz_path.name} {npz_path.stat().st_size / 2**30:.3f} GiB "
        f"elapsed={manifest['elapsed_seconds']}s",
        flush=True,
    )
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=base.DEFAULT_MODELS)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.shard_count <= 16:
        raise ValueError("--shard-count must be in [1,16]")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid --shard-index")

    args.output.mkdir(parents=True, exist_ok=True)
    dates = base.build_dates(args)
    config = {
        "spec": SPEC,
        "dates": {
            "count_this_shard": int(len(dates)),
            "first": dates[0].isoformat() if len(dates) else None,
            "last": dates[-1].isoformat() if len(dates) else None,
        },
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "lead_hours": list(base.TARGET_LEADS),
        "latitude_orientation": "north_to_south",
        "era5_source": ERA5_SOURCE,
        "humidity_contract": "WB2 fraction times 100 -> FuXi percent",
        "read_scheduler": "single-threaded",
        "model_files": {
            stage: {
                "path": str(args.model_dir / f"{stage}.onnx"),
                "sha256": base.sha256(args.model_dir / f"{stage}.onnx"),
            }
            for stage, _count in base.STAGES
        },
        "script_sha256": base.sha256(Path(__file__)),
        "base_script_sha256": base.sha256(Path(base.__file__)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    base.atomic_json(args.output / f"_run_config_shard{args.shard_index}.json", config)

    runner = FuXiModelAWB2(args.model_dir)
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
    base.atomic_json(args.output / f"_summary_shard{args.shard_index}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
