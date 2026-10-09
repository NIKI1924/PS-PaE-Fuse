#!/usr/bin/env python3
"""Deployable Phase-Selective PaE-Fuse (PS-PaE-Fuse).

The model combines a global-skill expert with an independently trained
phase-aware expert.  A frozen differentiable gate uses only the two forecasts'
predicted anomaly magnitudes.  It activates the phase expert when (i) either
expert forecasts an event-scale anomaly and (ii) the phase expert predicts the
more extreme magnitude.  ERA5 or any other verification target is never used in
forward inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from paefuse import PaEFuseNet
from wenqiong_v3_spf_dem import WenqiongV3SPFDEM


TRAIN_CLIMATOLOGY = (284.86, 0.04, 0.19, 101051.80, 55100.46, 279.33)
L3_EVENT_SCALES = (8.0, 8.0, 7.0, 1200.0, 3000.0, 7.0)


def common_kwargs() -> dict[str, object]:
    return dict(
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
    )


class PhaseSelectivePaEFuse(nn.Module):
    """Black-box dual-expert forecast fusion with a phase-selective gate."""

    def __init__(
        self,
        skill_expert: nn.Module,
        phase_expert: nn.Module,
        event_slope: float = 12.0,
        advantage_slope: float = 80.0,
        freeze_experts: bool = True,
    ) -> None:
        super().__init__()
        self.skill_expert = skill_expert
        self.phase_expert = phase_expert
        self.event_slope = float(event_slope)
        self.advantage_slope = float(advantage_slope)
        self.register_buffer(
            "training_climatology",
            torch.tensor(TRAIN_CLIMATOLOGY, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "event_scales",
            torch.tensor(L3_EVENT_SCALES, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.last_gate: torch.Tensor | None = None
        if freeze_experts:
            for parameter in self.skill_expert.parameters():
                parameter.requires_grad = False
            for parameter in self.phase_expert.parameters():
                parameter.requires_grad = False

    def forward(self, *args, **kwargs):
        skill_mu, skill_sigma, skill_logits = self.skill_expert(*args, **kwargs)
        phase_mu, phase_sigma, phase_logits = self.phase_expert(*args, **kwargs)
        climatology = self.training_climatology.to(skill_mu)
        scales = self.event_scales.to(skill_mu)
        skill_anomaly = (skill_mu - climatology).abs()
        phase_anomaly = (phase_mu - climatology).abs()
        maximum_normalized_anomaly = torch.maximum(skill_anomaly, phase_anomaly) / scales
        normalized_phase_advantage = (phase_anomaly - skill_anomaly) / scales
        event_gate = torch.sigmoid(
            self.event_slope * (maximum_normalized_anomaly - 1.0)
        )
        advantage_gate = torch.sigmoid(
            self.advantage_slope * normalized_phase_advantage
        )
        gate = event_gate * advantage_gate
        mu = skill_mu + gate * (phase_mu - skill_mu)
        sigma = skill_sigma + gate * (phase_sigma - skill_sigma)
        batch, variables, height, width = gate.shape
        logit_gate = gate.unsqueeze(2).expand(batch, variables, 3, height, width)
        skill_logits = skill_logits.reshape(batch, variables, 3, height, width)
        phase_logits = phase_logits.reshape(batch, variables, 3, height, width)
        logits = skill_logits + logit_gate * (phase_logits - skill_logits)
        self.last_gate = gate
        return mu, sigma, logits.reshape(batch, variables * 3, height, width)


def load_phase_selective_model(
    skill_checkpoint: str,
    phase_checkpoint: str,
    device: torch.device | str = "cpu",
) -> PhaseSelectivePaEFuse:
    kwargs = common_kwargs()
    skill = WenqiongV3SPFDEM(**kwargs)
    phase = PaEFuseNet(variant="full", **kwargs)
    skill.load_state_dict(
        torch.load(skill_checkpoint, map_location="cpu", weights_only=True)
    )
    phase.load_state_dict(
        torch.load(phase_checkpoint, map_location="cpu", weights_only=True)
    )
    model = PhaseSelectivePaEFuse(skill, phase, freeze_experts=True)
    return model.to(device).eval()


if __name__ == "__main__":
    model = load_phase_selective_model(
        "/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt",
        "/vol2/xrx/paefuse_pr_20260904/full_s123/best_v3_spf.pt",
        "cuda:0",
    )
    x = torch.randn(2, 24, 96, 96, device="cuda:0")
    lead = torch.tensor([2, 6], dtype=torch.long, device="cuda:0")
    lat0 = torch.tensor([100, 400], dtype=torch.long, device="cuda:0")
    lon0 = torch.tensor([300, 1200], dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        outputs = model(x, x, lead, lat0=lat0, lon0=lon0)
    print([tuple(value.shape) for value in outputs])
    print("mean gate", model.last_gate.mean(dim=(0, 2, 3)).tolist())
    print("PS-PaE-Fuse smoke test passed")
