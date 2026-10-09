#!/usr/bin/env python3
"""PaE-Fuse: phase-aware extreme-conditioned multi-model forecast fusion.

The model extends the established Wenqiong probabilistic fusion backbone with
an inference-time complex spectral router.  Unlike a phase-aware loss alone,
the router uses cross-member amplitude and phase coherence to select members
at each resolved frequency.  An extreme-risk signal from the jointly trained
classification head conditions the spatial/spectral correction mixture.

Input
-----
x_norm : (B, 4*6, H, W) normalized forecasts from four named members
x_raw  : (B, 4*6, H, W) forecasts in physical units

Output
------
mu, sigma, cls_logits : calibrated mean, Gaussian scale, and event logits
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wenqiong_dualtask import N_MODELS, N_VARS
from wenqiong_v3_spf_dem import WenqiongV3SPFDEM


PAEFUSE_VARIANTS = {
    "full",
    "amplitude_only",
    "no_extreme_condition",
    "phase_router_only",
}


class PhaseCoherentFrequencyRouter(nn.Module):
    """Route heterogeneous forecast members in the complex Fourier domain.

    The router never uses verifying truth.  Per-frequency scores are inferred
    from relative amplitude, phase alignment to the complex member consensus,
    cross-member disagreement, radial frequency, lead time, and member/variable
    embeddings.  Real softmax weights mix complex member coefficients before
    inverse transformation.
    """

    def __init__(
        self,
        patch_h: int = 96,
        patch_w: int = 96,
        pool_factor: int = 2,
        lead_dim: int = 4,
        identity_dim: int = 4,
        hidden_dim: int = 16,
        use_phase: bool = True,
    ) -> None:
        super().__init__()
        if patch_h % pool_factor or patch_w % pool_factor:
            raise ValueError("patch dimensions must be divisible by pool_factor")
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.pool_factor = pool_factor
        self.low_h = patch_h // pool_factor
        self.low_w = patch_w // pool_factor
        self.use_phase = use_phase

        self.lead_embed = nn.Embedding(7, lead_dim)
        self.identity_embed = nn.Parameter(
            torch.zeros(N_MODELS, N_VARS, identity_dim)
        )
        # amplitude z-score, phase cos/sin, disagreement, radial frequency
        feature_dim = 5 + lead_dim + identity_dim
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.identity_embed, std=0.02)
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)

        fy = torch.fft.fftfreq(self.low_h).view(self.low_h, 1)
        fx = torch.fft.rfftfreq(self.low_w).view(1, self.low_w // 2 + 1)
        radial = torch.sqrt(fy.square() + fx.square())
        radial = radial / radial.max().clamp_min(1e-6)
        self.register_buffer("radial_frequency", radial.float())

        self.last_weights: torch.Tensor | None = None
        self.last_coherence: torch.Tensor | None = None

    def forward(
        self,
        members_norm: torch.Tensor,
        lead_hour_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return spectral correction, member-frequency weights, coherence.

        members_norm has shape (B, M, V, H, W).  The correction and coherence
        maps have shape (B, V, H, W).
        """
        b, m, v, h, w = members_norm.shape
        if (m, v, h, w) != (N_MODELS, N_VARS, self.patch_h, self.patch_w):
            raise ValueError(
                f"expected (*,{N_MODELS},{N_VARS},{self.patch_h},{self.patch_w}), "
                f"got {tuple(members_norm.shape)}"
            )

        low = F.avg_pool2d(
            members_norm.reshape(b, m * v, h, w).float(),
            self.pool_factor,
            self.pool_factor,
        ).reshape(b, m, v, self.low_h, self.low_w)
        spectrum = torch.fft.rfft2(low, norm="ortho")
        amplitude = spectrum.abs().clamp_min(1e-6)
        log_amplitude = torch.log1p(amplitude)
        amp_mean = log_amplitude.mean(dim=1, keepdim=True)
        amp_std = log_amplitude.std(dim=1, keepdim=True).clamp_min(1e-4)
        relative_amplitude = (log_amplitude - amp_mean) / amp_std

        consensus = spectrum.mean(dim=1, keepdim=True)
        if self.use_phase:
            unit = spectrum / amplitude
            consensus_unit = consensus / consensus.abs().clamp_min(1e-6)
            relative_phase = unit * consensus_unit.conj()
            phase_cos = relative_phase.real
            phase_sin = relative_phase.imag
            disagreement = (
                (spectrum - consensus).abs()
                / amplitude.mean(dim=1, keepdim=True).clamp_min(1e-5)
            ).tanh()
        else:
            phase_cos = torch.zeros_like(relative_amplitude)
            phase_sin = torch.zeros_like(relative_amplitude)
            disagreement = (
                (amplitude - amplitude.mean(dim=1, keepdim=True)).abs()
                / amplitude.mean(dim=1, keepdim=True).clamp_min(1e-5)
            ).tanh()

        radial = self.radial_frequency.view(
            1, 1, 1, self.low_h, self.low_w // 2 + 1
        ).expand(b, m, v, -1, -1)
        base_features = torch.stack(
            [relative_amplitude, phase_cos, phase_sin, disagreement, radial],
            dim=-1,
        )

        lead = self.lead_embed(lead_hour_idx).view(b, 1, 1, 1, 1, -1)
        lead = lead.expand(b, m, v, self.low_h, self.low_w // 2 + 1, -1)
        identity = self.identity_embed.view(1, m, v, 1, 1, -1)
        identity = identity.expand(b, m, v, self.low_h, self.low_w // 2 + 1, -1)
        score_input = torch.cat([base_features, lead, identity], dim=-1)
        logits = self.score(score_input).squeeze(-1)
        weights = torch.softmax(logits, dim=1)

        mixed_spectrum = (weights * spectrum).sum(dim=1)
        mixed_low = torch.fft.irfft2(
            mixed_spectrum,
            s=(self.low_h, self.low_w),
            norm="ortho",
        )
        consensus_low = low.mean(dim=1)
        correction_low = mixed_low - consensus_low
        correction = F.interpolate(
            correction_low,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # One coherence reliability value per sample/variable, broadcast as a
        # spatial conditioning map.  It is descriptive, not a target-derived
        # signal, and therefore remains valid at inference.
        unit_all = spectrum / amplitude
        coherence_scalar = unit_all.mean(dim=1).abs().mean(
            dim=(-2, -1), keepdim=True
        )
        coherence = coherence_scalar.expand(b, v, h, w)
        self.last_weights = weights
        self.last_coherence = coherence_scalar
        return correction, weights, coherence


class PaEFuseNet(WenqiongV3SPFDEM):
    """Phase-aware, extreme-conditioned probabilistic fusion network."""

    def __init__(
        self,
        variant: str = "full",
        spectral_pool_factor: int = 2,
        phase_hidden_dim: int = 16,
        **kwargs,
    ) -> None:
        if variant not in PAEFUSE_VARIANTS:
            raise ValueError(f"unknown PaE-Fuse variant: {variant}")
        d_interact = int(kwargs.get("d_interact", 48))
        super().__init__(**kwargs)
        self.variant = variant
        self.phase_router = PhaseCoherentFrequencyRouter(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            pool_factor=spectral_pool_factor,
            hidden_dim=phase_hidden_dim,
            use_phase=(variant != "amplitude_only"),
        )
        gate_in = d_interact + 4 * N_VARS
        # interact + spread + phase correction magnitude + coherence + risk
        self.phase_gate = nn.Sequential(
            nn.Conv2d(gate_in, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, N_VARS, 1),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.phase_gate[-2].weight, std=0.01)
        nn.init.constant_(self.phase_gate[-2].bias, -2.0)

        self.last_spatial_routing_weights: torch.Tensor | None = None
        self.last_frequency_routing_weights: torch.Tensor | None = None
        self.last_phase_gate: torch.Tensor | None = None
        self.last_extreme_risk: torch.Tensor | None = None

    def forward(
        self,
        x_norm: torch.Tensor,
        x_raw: torch.Tensor | None = None,
        lead_hour_idx: torch.Tensor | None = None,
        lat0: torch.Tensor | None = None,
        lon0: torch.Tensor | None = None,
        flip_flag: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lat0 is None or lon0 is None or lead_hour_idx is None:
            raise ValueError("PaE-Fuse requires lat0, lon0, and lead_hour_idx")
        b, _, h, w = x_norm.shape
        members_norm = x_norm.reshape(b, N_MODELS, N_VARS, h, w)
        consensus_norm = members_norm.mean(dim=1)
        spread = members_norm.std(dim=1)
        deviations = members_norm - consensus_norm.unsqueeze(1)
        deviations_flat = deviations.reshape(b, -1, h, w)
        if x_raw is None:
            consensus_raw = consensus_norm
        else:
            consensus_raw = x_raw.reshape(
                b, N_MODELS, N_VARS, h, w
            ).mean(dim=1)

        interact = self.encoder_interact(deviations_flat, spread)
        if self.use_context:
            static_patch = self._slice_static_patch(lat0, lon0, flip_flag)
            context = self.context(torch.cat([x_norm, static_patch], dim=1))
            encoder_features = torch.cat([interact, context], dim=1)
        else:
            encoder_features = interact
        spatial_correction, spatial_weights = self.encoder_router(
            encoder_features,
            deviations_flat,
            lead_hour_idx,
        )

        phase_correction, frequency_weights, coherence = self.phase_router(
            members_norm,
            lead_hour_idx,
        )
        sphere_features = self.sph(lat0, lon0, lead_hour_idx, flip_flag)

        # Classification is evaluated before regression so its learned event
        # risk can condition the phase/spatial fusion at inference.
        cls_base = self.cls_refinement(spread, interact, deviations)
        cls_logits = self.cls_head(sphere_features, cls_base)
        extreme_risk = torch.sigmoid(
            cls_logits.reshape(b, N_VARS, self.n_thresholds, h, w)
        ).amax(dim=2)
        if self.variant == "no_extreme_condition":
            extreme_condition = torch.zeros_like(extreme_risk)
        else:
            extreme_condition = extreme_risk

        gate_input = torch.cat(
            [
                interact,
                spread,
                phase_correction.abs(),
                coherence,
                extreme_condition,
            ],
            dim=1,
        )
        phase_gate = self.phase_gate(gate_input)
        mixed_correction = (
            (1.0 - phase_gate) * spatial_correction
            + phase_gate * phase_correction
        )

        refined = self.reg_refinement(
            mixed_correction,
            interact,
            members_norm,
            spread,
        )
        mu = self.reg_head(sphere_features, refined, consensus_raw)
        sigma = self.sigma_head(spread, sphere_features)

        self.last_routing_weights = spatial_weights
        self.last_spatial_routing_weights = spatial_weights
        self.last_frequency_routing_weights = frequency_weights
        self.last_phase_gate = phase_gate
        self.last_extreme_risk = extreme_risk
        self.last_cls_logits = cls_logits
        return mu, sigma, cls_logits

    def param_summary(self) -> dict[str, int]:
        result = super().param_summary()
        result["phase_router"] = sum(p.numel() for p in self.phase_router.parameters())
        result["phase_gate"] = sum(p.numel() for p in self.phase_gate.parameters())
        result["total"] = sum(p.numel() for p in self.parameters())
        return result


def build_paefuse(variant: str = "full", **kwargs) -> PaEFuseNet:
    """Factory retained for training/evaluation scripts and checkpoints."""
    return PaEFuseNet(variant=variant, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PaEFuseNet(
        variant="full",
        static_dir="/home/xrx/wenqiong/supplement_20260904/static_fields",
        d_interact=48,
        d_route=32,
        base_ch=32,
        dropout=0.1,
        n_thresholds=3,
        sph_L_max=4,
        sph_n_lat_rbf=8,
        sph_d_out=32,
        ctx_d=16,
        use_context=True,
    ).eval()
    x = torch.randn(2, N_MODELS * N_VARS, 96, 96)
    lead = torch.tensor([2, 6], dtype=torch.long)
    lat0 = torch.tensor([100, 400], dtype=torch.long)
    lon0 = torch.tensor([300, 1200], dtype=torch.long)
    with torch.no_grad():
        mu, sigma, logits = model(x, x, lead, lat0=lat0, lon0=lon0)
    print(model.param_summary())
    print(mu.shape, sigma.shape, logits.shape)
    print("frequency weights", model.last_frequency_routing_weights.shape)
    print("phase gate", model.last_phase_gate.shape)
    print("PaE-Fuse smoke test passed")
