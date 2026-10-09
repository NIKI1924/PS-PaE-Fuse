#!/usr/bin/env python
"""train_v3_spfdem.py — Train v3 SPF with DEM/LSM/std_oro static channels.

Usage:
  python train_v3_spfdem.py --cache-dir ... --output-dir ... --gpu N \
      --epochs 35 --var-scale-z500 900 --seed 42   # combine zs9 + DEM + seed
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-parse --seed and apply BEFORE any torch import
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--seed', type=int, default=None)
_custom, _rest = _pre.parse_known_args()
if _custom.seed is not None:
    import random
    import numpy as np
    import torch
    random.seed(_custom.seed)
    np.random.seed(_custom.seed)
    torch.manual_seed(_custom.seed)
    torch.cuda.manual_seed_all(_custom.seed)
    torch.backends.cudnn.deterministic = False  # speed > strict determinism
    print(f'[train_v3_spfdem] seed set to {_custom.seed}', flush=True)
    sys.argv = [sys.argv[0]] + _rest  # strip --seed before train_v3_spf reads

# Monkey-patch BEFORE importing train_v3_spf
from wenqiong_v3_spf_dem import patch_in
patch_in()

import train_v3_spf

if __name__ == '__main__':
    train_v3_spf.main()
