#!/usr/bin/env python3
"""48-GiB-safe GraphCast Model-A launcher.

The memory-safe v2 launcher keeps host RAM bounded.  This final shim also
forces one autoregressive step per rollout chunk, reducing the observed GPU
peak that narrowly exceeded a 48-GiB device with two-step chunks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MAIN = Path("/home/xrx/wenqiong/data_rebuild/run_graphcast_modela_memorysafe_v2.py")
SECONDARY = Path(
    "/mnt/a/weather-ZJJ/wenqiong/data_rebuild/run_graphcast_modela_memorysafe_v2.py"
)
SOURCE = MAIN if MAIN.exists() else SECONDARY
spec = importlib.util.spec_from_file_location("graphcast_modela_memorysafe_v2", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load GraphCast v2 launcher: {SOURCE}")
implementation_v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = implementation_v2
spec.loader.exec_module(implementation_v2)

rollout_module = implementation_v2.implementation.base.rollout
_original_generator = rollout_module.chunked_prediction_generator


def _one_step_generator(*args, **kwargs):
    kwargs["num_steps_per_chunk"] = 1
    return _original_generator(*args, **kwargs)


rollout_module.chunked_prediction_generator = _one_step_generator

if __name__ == "__main__":
    raise SystemExit(implementation_v2.implementation.base.main())
