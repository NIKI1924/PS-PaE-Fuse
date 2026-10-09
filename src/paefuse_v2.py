#!/usr/bin/env python3
"""PaE-Fuse v2: conservative phase residual anchored to spatial routing.

Version 1 allowed the complex router to replace part of the established spatial
correction.  Its three-seed audit showed unstable member preferences and no
repeatable CSI gain.  Version 2 makes the phase path an explicitly bounded
residual around the existing spatial router and structurally couples its gate to
the already calibrated event-risk head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from paefuse import PaEFuseNet
from wenqiong_dualtask import N_MODELS, N_VARS


class PhaseAnchoredFrequencyRouter(nn.Module):
    """Bounded phase modulation of the spatial router's member prior."""

    def __init__(
        self,
        patch_h: int = 96,
        patch_w: int = 96,
        pool_factor: int = 2,
        lead_dim: int = 4,
        identity_dim: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.pool_factor = pool_factor
        self.low_h = patch_h // pool_factor
        self.low_w = patch_w // pool_factor
        self.lead_embed = nn.Embedding(7, lead_dim)
        self.identity_embed = nn.Parameter(torch.zeros(N_MODELS, N_VARS, identity_dim))
        self.score = nn.Sequential(
            nn.Linear(5 + lead_dim + identity_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Bounded phase logit perturbation: temperature in (0, 0.5).
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.identity_embed, std=0.02)
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)

        fy = torch.fft.fftfreq(self.low_h).view(self.low_h, 1)
        fx = torch.fft.rfftfreq(self.low_w).view(1, self.low_w // 2 + 1)
        radial = torch.sqrt(fy.square() + fx.square())
        radial = radial / radial.max().clamp_min(1e-6)
        self.register_buffer("radial_frequency", radial.float())
        self.last_weights: torch.Tensor | None = None
        self.last_prior: torch.Tensor | None = None
        self.last_coherence: torch.Tensor | None = None

    def forward(
        self,
        members_norm: torch.Tensor,
        lead_hour_idx: torch.Tensor,
        spatial_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, members, variables, height, width = members_norm.shape
        low = F.avg_pool2d(
            members_norm.reshape(batch, members * variables, height, width).float(),
            self.pool_factor,
            self.pool_factor,
        ).reshape(batch, members, variables, self.low_h, self.low_w)
        spectrum = torch.fft.rfft2(low, norm="ortho")
        amplitude = spectrum.abs().clamp_min(1e-6)
        log_amplitude = torch.log1p(amplitude)
        relative_amplitude = (
            log_amplitude - log_amplitude.mean(dim=1, keepdim=True)
        ) / log_amplitude.std(dim=1, keepdim=True).clamp_min(1e-4)

        consensus = spectrum.mean(dim=1, keepdim=True)
        unit = spectrum / amplitude
        consensus_unit = consensus / consensus.abs().clamp_min(1e-6)
        relative_phase = unit * consensus_unit.conj()
        disagreement = (
            (spectrum - consensus).abs()
            / amplitude.mean(dim=1, keepdim=True).clamp_min(1e-5)
        ).tanh()
        radial = self.radial_frequency.view(1, 1, 1, self.low_h, -1).expand(
            batch, members, variables, -1, -1
        )
        frequency_features = torch.stack(
            [
                relative_amplitude,
                relative_phase.real,
                relative_phase.imag,
                disagreement,
                radial,
            ],
            dim=-1,
        )
        lead = self.lead_embed(lead_hour_idx).view(batch, 1, 1, 1, 1, -1).expand(
            batch, members, variables, self.low_h, self.low_w // 2 + 1, -1
        )
        identity = self.identity_embed.view(1, members, variables, 1, 1, -1).expand(
            batch, members, variables, self.low_h, self.low_w // 2 + 1, -1
        )
        phase_score = torch.tanh(
            self.score(torch.cat([frequency_features, lead, identity], dim=-1)).squeeze(-1)
        )

        # The spatial router is the stable prior.  Phase can only make a bounded
        # logit adjustment rather than learning an unconstrained alternative.
        prior = spatial_weights.mean(dim=(-2, -1)).clamp_min(1e-4)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-6)
        prior_logits = prior.log().unsqueeze(-1).unsqueeze(-1)
        temperature = 0.5 * torch.sigmoid(self.temperature_raw)
        weights = torch.softmax(prior_logits + temperature * phase_score, dim=1)

        prior_spectrum = (prior.unsqueeze(-1).unsqueeze(-1) * spectrum).sum(dim=1)
        phase_spectrum = (weights * spectrum).sum(dim=1)
        residual_low = torch.fft.irfft2(
            phase_spectrum - prior_spectrum,
            s=(self.low_h, self.low_w),
            norm="ortho",
        )
        residual = F.interpolate(
            residual_low, size=(height, width), mode="bilinear", align_corners=False
        )
        coherence_scalar = unit.mean(dim=1).abs().mean(dim=(-2, -1), keepdim=True)
        coherence = coherence_scalar.expand(batch, variables, height, width)
        self.last_weights = weights
        self.last_prior = prior
        self.last_coherence = coherence_scalar
        return residual, weights, coherence


class PaEFuseV2(PaEFuseNet):
    """Warm-startable, bounded residual phase-conditioning model."""

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="full", **kwargs)
        self.phase_router = PhaseAnchoredFrequencyRouter(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            pool_factor=2,
            hidden_dim=16,
        )
        # Max absolute contribution is 0.25 times the gated phase residual.
        initial_gain_raw = math.atanh(0.08)  # 0.25*tanh(.) = 0.02
        self.residual_gain_raw = nn.Parameter(
            torch.full((N_VARS,), float(initial_gain_raw))
        )
        self._backbone_frozen = False

    def freeze_backbone(self) -> None:
        trainable_prefixes = ("phase_router.", "phase_gate.", "residual_gain_raw")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)
        self._backbone_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._backbone_frozen:
            for name, module in self.named_children():
                if name not in {"phase_router", "phase_gate"}:
                    module.eval()
            self.phase_router.train(True)
            self.phase_gate.train(True)
        return self

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
            raise ValueError("PaE-Fuse v2 requires lat0, lon0, and lead_hour_idx")
        batch, _, height, width = x_norm.shape
        members_norm = x_norm.reshape(batch, N_MODELS, N_VARS, height, width)
        consensus_norm = members_norm.mean(dim=1)
        spread = members_norm.std(dim=1)
        deviations = members_norm - consensus_norm.unsqueeze(1)
        deviations_flat = deviations.reshape(batch, -1, height, width)
        consensus_raw = (
            consensus_norm
            if x_raw is None
            else x_raw.reshape(batch, N_MODELS, N_VARS, height, width).mean(dim=1)
        )

        interact = self.encoder_interact(deviations_flat, spread)
        if self.use_context:
            static_patch = self._slice_static_patch(lat0, lon0, flip_flag)
            context = self.context(torch.cat([x_norm, static_patch], dim=1))
            encoder_features = torch.cat([interact, context], dim=1)
        else:
            encoder_features = interact
        spatial_correction, spatial_weights = self.encoder_router(
            encoder_features, deviations_flat, lead_hour_idx
        )
        phase_residual, frequency_weights, coherence = self.phase_router(
            members_norm, lead_hour_idx, spatial_weights
        )

        sphere_features = self.sph(lat0, lon0, lead_hour_idx, flip_flag)
        cls_base = self.cls_refinement(spread, interact, deviations)
        cls_logits = self.cls_head(sphere_features, cls_base)
        extreme_risk = torch.sigmoid(
            cls_logits.reshape(batch, N_VARS, self.n_thresholds, height, width)
        ).amax(dim=2)
        gate_input = torch.cat(
            [interact, spread, phase_residual.abs(), coherence, extreme_risk], dim=1
        )
        learned_gate = self.phase_gate(gate_input)
        # Explicit risk coupling prevents the new path from being equally active
        # in ordinary and event pixels when the 1x1 gate ignores its risk input.
        risk_gate = learned_gate * (0.20 + 0.80 * extreme_risk)
        gain = 0.25 * torch.tanh(self.residual_gain_raw).view(1, -1, 1, 1)
        correction = spatial_correction + gain * risk_gate * phase_residual

        refined = self.reg_refinement(correction, interact, members_norm, spread)
        mu = self.reg_head(sphere_features, refined, consensus_raw)
        sigma = self.sigma_head(spread, sphere_features)
        self.last_routing_weights = spatial_weights
        self.last_spatial_routing_weights = spatial_weights
        self.last_frequency_routing_weights = frequency_weights
        self.last_phase_gate = risk_gate
        self.last_extreme_risk = extreme_risk
        self.last_cls_logits = cls_logits
        return mu, sigma, cls_logits

    def param_summary(self) -> dict[str, int]:
        result = super().param_summary()
        result["phase_router"] = sum(p.numel() for p in self.phase_router.parameters())
        result["phase_gate"] = sum(p.numel() for p in self.phase_gate.parameters())
        result["phase_residual_gain"] = self.residual_gain_raw.numel()
        result["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result["total"] = sum(p.numel() for p in self.parameters())
        return result
