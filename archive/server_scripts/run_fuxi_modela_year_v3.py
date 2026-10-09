#!/usr/bin/env python3
"""FuXi Model-A v3: robust, audited clipping of WB2 relative humidity.

WeatherBench2 stores derived relative humidity as a fraction, but the derived
field is not physically clipped and can be below 0 or above 1 at a small
number of grid cells (especially upper-air dry points).  FuXi expects percent
in [0, 100].  V3 therefore:

1. verifies that the field is finite;
2. records the raw range and the exact low/high clipping counts;
3. clips the fraction to [0, 1] and converts to percent.

Completed v2 artifacts are retained because v2 used the same clipping and
conversion after its overly strict range assertion.  Only missing dates are
computed by v3.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd

import run_fuxi_modela_year as base
import run_fuxi_modela_year_v2 as v2


SPEC = "wenqiong-modela-fuxi-v3-wb2era5-rhclip"
ACCEPTED_EXISTING_SPECS = {v2.SPEC, SPEC}


class FuXiModelAWB2RHClip(v2.FuXiModelAWB2):
    def build_input(self, init_time: pd.Timestamp) -> tuple[np.ndarray, dict]:
        input_times = [init_time - pd.Timedelta(hours=6), init_time]
        started = time.time()
        pressure = self._load_group(v2.PRESSURE_VARIABLES, input_times, levels=True)
        surface = self._load_group(v2.SURFACE_VARIABLES, input_times, levels=False)
        try:
            rh_fraction = pressure.relative_humidity.values.astype(np.float32)
            if not np.isfinite(rh_fraction).all():
                raise ValueError(
                    "WB2 relative_humidity contains NaN/Inf: "
                    f"finite_fraction={np.isfinite(rh_fraction).mean():.8f}"
                )
            rh_min = float(rh_fraction.min())
            rh_max = float(rh_fraction.max())
            low_count = int(np.count_nonzero(rh_fraction < 0.0))
            high_count = int(np.count_nonzero(rh_fraction > 1.0))
            total_count = int(rh_fraction.size)
            relative_humidity_percent = (
                np.clip(rh_fraction, 0.0, 1.0) * 100.0
            ).astype(np.float32)

            output = np.empty(
                (2, 70, base.N_LAT, base.N_LON), dtype=np.float32
            )
            output[:, 0:13] = pressure.geopotential.values.astype(np.float32)
            output[:, 13:26] = pressure.temperature.values.astype(np.float32)
            output[:, 26:39] = (
                pressure.u_component_of_wind.values.astype(np.float32)
            )
            output[:, 39:52] = (
                pressure.v_component_of_wind.values.astype(np.float32)
            )
            output[:, 52:65] = relative_humidity_percent
            output[:, 65] = surface["2m_temperature"].values.astype(np.float32)
            output[:, 66] = (
                surface["10m_u_component_of_wind"].values.astype(np.float32)
            )
            output[:, 67] = (
                surface["10m_v_component_of_wind"].values.astype(np.float32)
            )
            output[:, 68] = (
                surface["mean_sea_level_pressure"].values.astype(np.float32)
            )
            output[:, 69] = (
                surface["total_precipitation_6hr"].values.astype(np.float32)
            )
        finally:
            pressure.close()
            surface.close()

        if not np.isfinite(output).all():
            raise ValueError(
                "FuXi input has NaN/Inf: "
                f"finite_fraction={np.isfinite(output).mean():.8f}"
            )
        details = {
            "input_times": [timestamp.isoformat() for timestamp in input_times],
            "humidity_source": "WB2 ERA5 derived relative_humidity fraction",
            "humidity_contract": "finite -> clip [0,1] -> multiply by 100",
            "humidity_fraction_range_before_clipping": [rh_min, rh_max],
            "humidity_clipping": {
                "below_zero_count": low_count,
                "above_one_count": high_count,
                "total_count": total_count,
                "below_zero_fraction": low_count / total_count,
                "above_one_fraction": high_count / total_count,
            },
            "precipitation_source": "WB2 ERA5 total_precipitation_6hr",
            "read_scheduler": "single-threaded",
            "build_seconds": round(time.time() - started, 3),
            "summary": base.array_stats(output),
        }
        return output, details

    def run(self, init_time: pd.Timestamp):
        arrays, manifest = super().run(init_time)
        manifest["spec"] = SPEC
        return arrays, manifest


def valid_existing(npz_path: Path, manifest_path: Path) -> bool:
    if not npz_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("spec") not in ACCEPTED_EXISTING_SPECS
            or manifest.get("lead_hours") != list(base.TARGET_LEADS)
        ):
            return False
        with np.load(npz_path, allow_pickle=False) as archive:
            expected = {name for name, _channel in base.WQ_CHANNELS}
            return set(archive.files) == expected and all(
                archive[name].shape
                == (len(base.TARGET_LEADS), base.N_LAT, base.N_LON)
                for name in expected
            )
    except Exception:
        return False


def run_one(
    runner: FuXiModelAWB2RHClip,
    timestamp: pd.Timestamp,
    output_dir: Path,
) -> str:
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
        f"[{tag}] committed {npz_path.name} "
        f"{npz_path.stat().st_size / 2**30:.3f} GiB "
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
        "era5_source": v2.ERA5_SOURCE,
        "humidity_contract": "finite -> clip fraction [0,1] -> percent [0,100]",
        "accepted_existing_specs": sorted(ACCEPTED_EXISTING_SPECS),
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
        "v2_script_sha256": base.sha256(Path(v2.__file__)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    base.atomic_json(
        args.output / f"_run_config_v3_shard{args.shard_index}.json", config
    )

    runner = FuXiModelAWB2RHClip(args.model_dir)
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
                        f"[{timestamp.isoformat()}] attempt "
                        f"{attempt}/{args.max_attempts} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if attempt == args.max_attempts:
                        failures[timestamp.isoformat()] = (
                            f"{type(exc).__name__}: {exc}"
                        )
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
    base.atomic_json(
        args.output / f"_summary_v3_shard{args.shard_index}.json", summary
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
