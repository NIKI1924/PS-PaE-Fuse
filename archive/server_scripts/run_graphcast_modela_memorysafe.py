#!/usr/bin/env python3
"""Memory-safe launcher for ``run_graphcast_modela_year.py``.

The original, proven 2021 replay code represented 28 future target frames as
full NumPy NaN arrays.  At 37 levels this consumes more than 80 GiB while the
values are only a shape template.  GraphCast rollout supports lazy Dask target
templates, so this launcher replaces only batch construction and the small
run adapter while retaining the checkpoint, retry, output, manifest, and CLI
contract from the base script.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import os
import sys
import time
from pathlib import Path


MAIN_BASE = Path("/home/xrx/wenqiong/data_rebuild/run_graphcast_modela_year.py")
OLD_BASE = Path("/mnt/a/weather-ZJJ/wenqiong/data_rebuild/run_graphcast_modela_year.py")
BASE_SCRIPT = MAIN_BASE if MAIN_BASE.exists() else OLD_BASE

# The portable base script probes the secondary default eagerly.  On the main
# server, point it to the audited copy and satisfy only that harmless probe.
if MAIN_BASE.exists():
    os.environ["GC_STATICS"] = "/home/xrx/wenqiong/data_rebuild/graphcast_statics.npz"
    _original_exists = Path.exists

    def _portable_exists(path: Path) -> bool:
        if str(path) == "/home/deploy/gc/statics.npz":
            return True
        return _original_exists(path)

    Path.exists = _portable_exists

spec = importlib.util.spec_from_file_location("graphcast_modela_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402


def _lazy_future(actual: np.ndarray, time_axis: int) -> da.Array:
    """Append N_STEPS lazy NaN frames without materializing them in RAM."""
    actual = np.asarray(actual, dtype=np.float32)
    actual_chunks = list(actual.shape)
    actual_chunks[time_axis] = actual.shape[time_axis]
    actual_dask = da.from_array(actual, chunks=tuple(actual_chunks))
    future_shape = list(actual.shape)
    future_shape[time_axis] = base.N_STEPS
    future_chunks = list(future_shape)
    future_chunks[time_axis] = 2
    future = da.full(
        tuple(future_shape),
        np.nan,
        chunks=tuple(future_chunks),
        dtype=np.float32,
    )
    return da.concatenate([actual_dask, future], axis=time_axis)


def memory_safe_build_batch(self, init_time: pd.Timestamp):
    started = time.time()
    all_times = pd.date_range(
        init_time - pd.Timedelta(hours=6),
        periods=2 + base.N_STEPS,
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

    fields = {}
    with np.load(base.STATICS) as static_archive:
        surface_geopotential = static_archive["geopotential_at_surface"]
        land_sea_mask = static_archive["land_sea_mask"]
        static_latitude = static_archive["lat"]
        if bool(static_latitude[0] < static_latitude[-1]) != bool(
            latitude[0] < latitude[-1]
        ):
            surface_geopotential = surface_geopotential[::-1]
            land_sea_mask = land_sea_mask[::-1]
        fields["geopotential_at_surface"] = (
            ("lat", "lon"),
            surface_geopotential.astype(np.float32),
        )
        fields["land_sea_mask"] = (
            ("lat", "lon"),
            land_sea_mask.astype(np.float32),
        )

    def load_surface():
        with base.dask.config.set(scheduler="single-threaded"):
            return self.dataset[list(base.SURFACE)].sel(time=input_times).compute()

    surface = base.retry_read(load_surface, "surface")
    for variable in base.SURFACE:
        actual = orient(surface[variable].values)[None, ...]
        fields[variable] = (
            ("batch", "time", "lat", "lon"),
            _lazy_future(actual, time_axis=1),
        )
    surface.close()

    def load_toa():
        with base.dask.config.set(scheduler="single-threaded"):
            return self.dataset["toa_incident_solar_radiation"].sel(
                time=list(all_times)
            ).values

    toa = base.retry_read(load_toa, "toa forcing", timeout=600).astype(np.float32)
    fields["toa_incident_solar_radiation"] = (
        ("batch", "time", "lat", "lon"),
        orient(toa)[None, ...],
    )
    del toa

    precipitation_hours = pd.date_range(
        input_times[0] - pd.Timedelta(hours=5),
        input_times[1],
        freq="1h",
    )

    def load_precipitation():
        with base.dask.config.set(scheduler="single-threaded"):
            return self.dataset["total_precipitation"].sel(
                time=list(precipitation_hours)
            ).compute()

    precipitation = base.retry_read(load_precipitation, "precipitation")
    precipitation_actual = np.empty((1, 2, base.N_LAT, base.N_LON), np.float32)
    for index, timestamp in enumerate(input_times):
        window = pd.date_range(timestamp - pd.Timedelta(hours=5), timestamp, freq="1h")
        precipitation_actual[0, index] = orient(
            precipitation.sel(time=list(window)).sum("time").values.astype(np.float32)
        )
    fields["total_precipitation_6hr"] = (
        ("batch", "time", "lat", "lon"),
        _lazy_future(precipitation_actual, time_axis=1),
    )
    precipitation.close()

    for variable in base.ATMOSPHERIC:
        def load_atmosphere(variable_name=variable):
            with base.dask.config.set(scheduler="single-threaded"):
                return self.dataset[variable_name].sel(
                    time=input_times,
                    level=base.LEVELS,
                ).values

        values = base.retry_read(
            load_atmosphere,
            f"atmosphere:{variable}",
        ).astype(np.float32)
        actual = orient(values)[None, ...]
        fields[variable] = (
            ("batch", "time", "level", "lat", "lon"),
            _lazy_future(actual, time_axis=1),
        )
        del values, actual

    relative_time = (
        np.arange(2 + base.N_STEPS) * 6 * 3600 * 1_000_000_000
    ).astype("timedelta64[ns]")
    batch = xr.Dataset(
        fields,
        coords={
            "lon": ("lon", longitude),
            "lat": ("lat", latitude),
            "level": ("level", np.asarray(base.LEVELS, np.int32)),
            "time": ("time", relative_time),
            "datetime": (
                ("batch", "time"),
                np.asarray(all_times).reshape(1, 2 + base.N_STEPS),
            ),
        },
    )
    details = {
        "input_times": [timestamp.isoformat() for timestamp in input_times],
        "source_latitude_orientation": "north_to_south"
        if flip_to_south_north
        else "south_to_north",
        "graphcast_input_orientation": "south_to_north",
        "target_template": "lazy_dask_nan_chunks_time_2",
        "read_scheduler": "single-threaded",
        "build_seconds": round(time.time() - started, 3),
    }
    return batch, details


def memory_safe_run(self, init_time: pd.Timestamp):
    started = time.time()
    example_batch, input_details = self.build_batch(init_time)
    inputs, targets, forcings = base.data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{base.N_STEPS * 6}h"),
        **dataclasses.asdict(self.task_config),
    )
    # Materialize only two real input frames and the comparatively small
    # forcing fields.  Targets remain lazy and are computed two steps at a time
    # by GraphCast's chunked rollout implementation.
    inputs = inputs.compute()
    forcings = forcings.compute()

    accumulated = {name: [] for name in base.OUTPUT_NAMES}
    for chunk in base.rollout.chunked_prediction_generator(
        self.run_forward,
        rng=base.jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets,
        forcings=forcings,
        num_steps_per_chunk=2,
    ):
        for variable, level, output_name in base.WQ_OUTPUTS:
            data_array = chunk[variable].isel(batch=0)
            if level is not None:
                data_array = data_array.sel(level=level)
            accumulated[output_name].append(
                data_array.transpose("time", "lat", "lon").values.astype(np.float32)
            )
        del chunk

    arrays = {}
    for output_name in base.OUTPUT_NAMES:
        all_steps = np.concatenate(accumulated[output_name], axis=0)
        if all_steps.shape != (base.N_STEPS, base.N_LAT, base.N_LON):
            raise ValueError(f"{output_name}: bad rollout shape {all_steps.shape}")
        selected = all_steps[base.TARGET_INDICES, ::-1, :].copy()
        if not np.isfinite(selected).all():
            raise ValueError(
                f"{output_name}: non-finite fraction "
                f"{1.0 - float(np.isfinite(selected).mean()):.8f}"
            )
        arrays[output_name] = selected

    manifest = {
        "status": "complete",
        "spec": base.SPEC,
        "init_time": init_time.isoformat(),
        "lead_hours": list(base.TARGET_LEADS),
        "variables": sorted(base.OUTPUT_NAMES),
        "grid": {
            "latitude": "90_to_-90",
            "longitude": "0_to_359.75",
            "shape": [base.N_LAT, base.N_LON],
        },
        "checkpoint": {
            "path": str(base.PARAMS),
            "sha256": base.sha256(base.PARAMS),
            "training_period": "ERA5 1979-2017",
        },
        "era5_source": base.ARCO,
        "input": input_details,
        "output_summary": {
            name: base.array_stats(array) for name, array in arrays.items()
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    del example_batch, inputs, targets, forcings
    gc.collect()
    return arrays, manifest


base.GraphCastModelA.build_batch = memory_safe_build_batch
base.GraphCastModelA.run = memory_safe_run

if __name__ == "__main__":
    raise SystemExit(base.main())
