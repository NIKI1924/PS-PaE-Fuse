#!/usr/bin/env python3
"""Final launcher for memory-safe GraphCast Model-A inference.

This compatibility shim preserves the memory-safe implementation and adapts
the predictor callback to the keyword signature used by GraphCast rollout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MAIN = Path("/home/xrx/wenqiong/data_rebuild/run_graphcast_modela_memorysafe.py")
SECONDARY = Path(
    "/mnt/a/weather-ZJJ/wenqiong/data_rebuild/run_graphcast_modela_memorysafe.py"
)
SOURCE = MAIN if MAIN.exists() else SECONDARY
spec = importlib.util.spec_from_file_location("graphcast_modela_memorysafe", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load memory-safe GraphCast launcher: {SOURCE}")
implementation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = implementation
spec.loader.exec_module(implementation)

_original_init = implementation.base.GraphCastModelA.__init__


def _compatible_init(self) -> None:
    _original_init(self)
    original_forward = self.run_forward

    def run_forward(*, rng, inputs, targets_template, forcings):
        return original_forward(rng, inputs, targets_template, forcings)

    self.run_forward = run_forward


implementation.base.GraphCastModelA.__init__ = _compatible_init

if __name__ == "__main__":
    raise SystemExit(implementation.base.main())
