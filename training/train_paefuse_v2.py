#!/usr/bin/env python3
"""Warm-start and train the conservative PaE-Fuse v2 residual."""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--seed", type=int, default=123)
pre.add_argument(
    "--warm-start",
    default="/home/xrx/wenqiong/ov_ffl2_s123/best_v3_spf.pt",
)
pre.add_argument("--ffl-lambda", type=float, default=0.05)
pre.add_argument("--hp-lambda", type=float, default=0.10)
pre.add_argument("--ffl-gamma", type=float, default=1.0)
pre.add_argument("--unfreeze-backbone", action="store_true")
custom, remaining = pre.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

random.seed(custom.seed)
np.random.seed(custom.seed)
torch.manual_seed(custom.seed)
torch.cuda.manual_seed_all(custom.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from paefuse_v2 import PaEFuseV2
import wenqiong_v3_spf


def model_factory(**kwargs):
    kwargs.setdefault("static_dir", "/home/xrx/wenqiong/supplement_20260904/static_fields")
    model = PaEFuseV2(**kwargs)
    state = torch.load(custom.warm_start, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {
        name for name in model.state_dict()
        if name.startswith("phase_router.")
        or name.startswith("phase_gate.")
        or name == "residual_gain_raw"
    }
    nonexpected_missing = set(missing) - expected_missing
    if nonexpected_missing or unexpected:
        raise RuntimeError(
            f"warm-start mismatch missing={sorted(nonexpected_missing)} unexpected={unexpected}"
        )
    if not custom.unfreeze_backbone:
        model.freeze_backbone()
    print(
        f"[PaE-Fuse v2] seed={custom.seed} warm_start={custom.warm_start} "
        f"freeze_backbone={not custom.unfreeze_backbone} summary={model.param_summary()}",
        flush=True,
    )
    return model


wenqiong_v3_spf.WenqiongV3SPF = model_factory

import losses_v3

OriginalGaussianCRPS = losses_v3.LatWeightedGaussianCRPSLoss


def focal_frequency_loss(prediction, truth, gamma):
    forecast_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
    truth_spectrum = torch.fft.rfft2(truth.float(), norm="ortho")
    distance = (forecast_spectrum - truth_spectrum).abs().square()
    weight = distance.detach().pow(gamma)
    weight = weight / weight.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (weight * distance).mean()


def high_pass(x, kernel_size=9):
    return x - F.avg_pool2d(x, kernel_size, 1, kernel_size // 2)


class PhaseAugmentedCRPS(OriginalGaussianCRPS):
    def __init__(self, var_error_scale=None, use_lat_weight=True):
        super().__init__(var_error_scale=var_error_scale, use_lat_weight=use_lat_weight)
        self.step = 0

    def forward(self, mu, sigma, y, lat0=None):
        crps = super().forward(mu, sigma, y, lat0=lat0)
        scale = self.var_scale.view(1, -1, 1, 1).to(mu.device)
        prediction = mu / scale
        truth = y / scale
        ffl = focal_frequency_loss(prediction, truth, custom.ffl_gamma)
        hp = (high_pass(prediction) - high_pass(truth)).square().mean()
        total = crps + custom.ffl_lambda * ffl + custom.hp_lambda * hp
        if self.step % 200 == 0:
            print(
                f"[PaE-Fuse v2 loss] step={self.step} crps={crps.item():.5f} "
                f"ffl={ffl.item():.5f} hp={hp.item():.5f} total={total.item():.5f}",
                flush=True,
            )
        self.step += 1
        return total


losses_v3.LatWeightedGaussianCRPSLoss = PhaseAugmentedCRPS
import train_v3_spf

train_v3_spf.WenqiongV3SPF = model_factory
train_v3_spf.LatWeightedGaussianCRPSLoss = PhaseAugmentedCRPS


if __name__ == "__main__":
    train_v3_spf.main()
