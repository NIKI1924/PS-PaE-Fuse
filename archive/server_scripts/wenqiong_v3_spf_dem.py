#!/usr/bin/env python
"""wenqiong_v3_spf_dem.py — v3 SPF + 3 static channels (DEM/LSM/std_oro) via context branch.

Design (minimal disruption):
- Load global static buffer (3, 721, 1440) at __init__, normalized (z-score for oro/std_oro,
  raw 0-1 for LSM). Apply [::-1] lat flip (file is S→N, WB2 cache is N→S).
- In forward, slice static_patch (B, 3, pH, pW) using lat0/lon0 (mirrors SphereHyperNet).
- Replace MultiScaleContextBranch with in_ch=24+3=27 version, feed [x_norm, static_patch].
- Everything else (I2MoE, router, refinement, sphere, heads) unchanged.

Training/eval scripts call train_v3_spfdem.py / eval_v3_latband_dem.py which monkey-patch.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wenqiong_v3_spf import (
    WenqiongV3SPF, MultiScaleContextBranch,
    GRID_H, GRID_W, PATCH_H, PATCH_W,
)
from wenqiong_dualtask import N_VARS, N_MODELS


__all__ = ['WenqiongV3SPFDEM', 'patch_in', 'STATIC_DIR']

STATIC_DIR = '/tank2/weatherModels/MoE_zjj/MoE/results/unified/static_fields'


def _load_static_global(static_dir=STATIC_DIR, dtype=np.float32):
    """Load + normalize + flip 3 static channels to (3, 721, 1440) float32 (N->S)."""
    oro = np.load(os.path.join(static_dir, 'orography.npy'))[::-1].copy()
    lsm = np.load(os.path.join(static_dir, 'land_sea_mask.npy'))[::-1].copy()
    std_oro = np.load(os.path.join(static_dir, 'std_orography.npy'))[::-1].copy()
    # Z-score normalization for oro/std_oro (LSM is already 0-1)
    oro_z = (oro - oro.mean()) / (oro.std() + 1e-6)
    std_oro_z = (std_oro - std_oro.mean()) / (std_oro.std() + 1e-6)
    stack = np.stack([oro_z, lsm, std_oro_z], axis=0).astype(dtype)
    return stack  # (3, 721, 1440)


class MultiScaleContextBranchDEM(MultiScaleContextBranch):
    """Identical to parent, but defaults in_ch to N_VARS*N_MODELS + 3 = 27."""
    def __init__(self, in_ch=N_VARS * N_MODELS + 3, mid_ch=32, d_ctx=16, pool_factor=4):
        super().__init__(in_ch=in_ch, mid_ch=mid_ch, d_ctx=d_ctx, pool_factor=pool_factor)


class WenqiongV3SPFDEM(WenqiongV3SPF):
    """Wenqiong v3 SPF with 3 static channels (orography/LSM/std_orography) fed into context."""

    def __init__(self, static_dir=STATIC_DIR, n_static=3, **kwargs):
        super().__init__(**kwargs)
        self.n_static = n_static
        # Load static global buffer (~12 MB)
        static = _load_static_global(static_dir)
        self.register_buffer('static_global',
                             torch.from_numpy(static).contiguous())
        # Replace context branch with DEM-aware (in_ch = 24 + 3 = 27)
        if self.use_context:
            ctx_mid = 32
            ctx_d = kwargs.get('ctx_d', 16)
            self.context = MultiScaleContextBranchDEM(
                in_ch=N_VARS * N_MODELS + n_static,
                mid_ch=ctx_mid, d_ctx=ctx_d, pool_factor=4)
        else:
            print('  [WenqiongV3SPFDEM] WARNING: use_context=False, static fields unused')

    def _slice_static_patch(self, lat0, lon0, flip_flag=None):
        """Slice (B, 3, pH, pW) from static_global using lat0/lon0 (mirrors SphereHyperNet)."""
        B = lat0.shape[0]
        device = lat0.device
        pH, pW = self.patch_h, self.patch_w
        n_static = self.n_static

        h_off = torch.arange(pH, device=device, dtype=torch.long)
        w_off = torch.arange(pW, device=device, dtype=torch.long)

        if flip_flag is not None and flip_flag.any():
            flip_l = flip_flag.long()
            w_off_flip = (pW - 1 - w_off).unsqueeze(0).expand(B, -1)
            w_off_norm = w_off.unsqueeze(0).expand(B, -1)
            w_off_b = torch.where(flip_l.unsqueeze(1) > 0, w_off_flip, w_off_norm)
        else:
            w_off_b = w_off.unsqueeze(0).expand(B, -1)

        row_idx = lat0.unsqueeze(1) + h_off.unsqueeze(0)
        row_idx = row_idx.clamp(min=0, max=GRID_H - 1)
        col_idx = (lon0.unsqueeze(1) + w_off_b) % GRID_W

        row_exp = row_idx.unsqueeze(-1).expand(B, pH, pW)
        col_exp = col_idx.unsqueeze(1).expand(B, pH, pW)
        # static_global: (3, 721, 1440) → indexed: (3, B, pH, pW)
        static = self.static_global[:, row_exp, col_exp]
        return static.permute(1, 0, 2, 3).contiguous()  # (B, 3, pH, pW)

    def forward(self, x_norm, x_raw=None, lead_hour_idx=None,
                lat0=None, lon0=None, flip_flag=None):
        assert lat0 is not None and lon0 is not None, \
            "v3 SPF DEM requires lat0, lon0 for static field slicing"
        B, _, pH, pW = x_norm.shape

        # Feature engineering (same as parent)
        mn = x_norm.reshape(B, N_MODELS, N_VARS, pH, pW)
        consensus_n = mn.mean(dim=1)
        spread = mn.std(dim=1)
        dev_n = mn - consensus_n.unsqueeze(1)
        dev_flat = dev_n.reshape(B, -1, pH, pW)
        if x_raw is not None:
            consensus_raw = x_raw.reshape(B, N_MODELS, N_VARS, pH, pW).mean(dim=1)
        else:
            consensus_raw = consensus_n

        # Shared encoder
        interact = self.encoder_interact(dev_flat, spread)

        if self.use_context:
            static_patch = self._slice_static_patch(lat0, lon0, flip_flag)  # (B, 3, H, W)
            ctx_input = torch.cat([x_norm, static_patch], dim=1)  # (B, 27, H, W)
            ctx = self.context(ctx_input)
            enc_feat = torch.cat([interact, ctx], dim=1)
        else:
            enc_feat = interact

        correction, rw = self.encoder_router(enc_feat, dev_flat, lead_hour_idx)
        self.last_routing_weights = rw

        sph_feat = self.sph(lat0, lon0, lead_hour_idx, flip_flag)
        refined = self.reg_refinement(correction, interact, mn, spread)
        mu = self.reg_head(sph_feat, refined, consensus_raw)
        sigma = self.sigma_head(spread, sph_feat)
        cls_base = self.cls_refinement(spread, interact, dev_n)
        cls_logits = self.cls_head(sph_feat, cls_base)
        self.last_cls_logits = cls_logits

        return mu, sigma, cls_logits


def patch_in():
    """Monkey-patch wenqiong_v3_spf.WenqiongV3SPF → WenqiongV3SPFDEM.

    Call this BEFORE importing/running train_v3_spf.main().
    """
    import wenqiong_v3_spf
    wenqiong_v3_spf.WenqiongV3SPF = WenqiongV3SPFDEM


if __name__ == '__main__':
    # Smoke test
    print('Loading static global...')
    static = _load_static_global()
    print(f'  shape={static.shape} dtype={static.dtype}')
    print(f'  oro_z: mean={static[0].mean():.4f} std={static[0].std():.4f} min={static[0].min():.2f} max={static[0].max():.2f}')
    print(f'  lsm:   mean={static[1].mean():.4f} std={static[1].std():.4f} min={static[1].min():.2f} max={static[1].max():.2f}')
    print(f'  std_oro_z: mean={static[2].mean():.4f} std={static[2].std():.4f} min={static[2].min():.2f} max={static[2].max():.2f}')
    print('  AFTER [::-1] flip: row0 = N pole (low oro expected)')
    print(f'  row0[0].mean (oro_z N pole) = {static[0, 0].mean():.4f}  (should be < 0)')
    print(f'  row720[0].mean (oro_z S pole, Antarctica) = {static[0, 720].mean():.4f}  (should be > 0)')

    print('\nBuilding model...')
    model = WenqiongV3SPFDEM().eval()
    ps = model.param_summary()
    for k, v in ps.items():
        print(f'  {k}: {v:,}')

    B = 2
    x_norm = torch.randn(B, 24, PATCH_H, PATCH_W)
    x_raw = torch.randn(B, 24, PATCH_H, PATCH_W) * 10
    lead_idx = torch.tensor([0, 3], dtype=torch.long)
    lat0 = torch.tensor([100, 500], dtype=torch.long)
    lon0 = torch.tensor([300, 800], dtype=torch.long)
    with torch.no_grad():
        mu, sigma, cls = model(x_norm, x_raw, lead_idx, lat0=lat0, lon0=lon0)
    print(f'\nforward: mu {tuple(mu.shape)} sigma {tuple(sigma.shape)} cls {tuple(cls.shape)}')
    print('OK: DEM model smoke test passed.')
