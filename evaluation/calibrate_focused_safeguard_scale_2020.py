#!/usr/bin/env python3
"""Fit one frozen safeguard uncertainty multiplier on blocked 2020 data.

The predictive mean and safeguard gate are already frozen.  This script fits
only one scalar multiplying U10/V10 Gaussian scales where the forecast-only
safeguard gate is active.  Four contiguous date folds audit transfer, and the
final scalar is refitted on all 2020 dates before the 2022 seal is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/xrx/wenqiong")

from evaluate_extreme_safeguard_2020 import phase_selective_batch
from evaluate_paefuse_crossyear_2021 import load_thresholds
from evaluate_paefuse_patch import build
from unified_phase_scorecard import CACHE, CHECKPOINTS, CP, NORM, P, TC, latitude_weights


TARGET_LEAD_INDEX = (2, 4, 6)
LEAD_HOURS = {2: 72, 4: 120, 6: 168}
ROOT = Path("/vol2/xrx/paefuse_pr_20260904")
WORK = Path("/home/xrx/wenqiong/paefuse_pr_20260904")
SCALES = np.round(np.arange(0.50, 2.51, 0.10), 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_crps(
    mean: torch.Tensor, scale: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    scale = scale.clamp_min(0.02)
    z = (truth - mean) / scale
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return scale * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def safeguard_batch(
    base_mu: np.ndarray,
    raw: np.ndarray,
    trigger: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    members = raw.reshape(len(raw), 4, 6, P, P)
    member_speed = np.hypot(members[:, :, 1], members[:, :, 2])
    if config["source"] == "pangu":
        expert_u, expert_v = members[:, 0, 1], members[:, 0, 2]
        expert_speed = member_speed[:, 0]
    elif config["source"] == "max_member":
        index = member_speed.argmax(axis=1)
        expert_u = np.take_along_axis(members[:, :, 1], index[:, None], axis=1)[:, 0]
        expert_v = np.take_along_axis(members[:, :, 2], index[:, None], axis=1)[:, 0]
        expert_speed = member_speed.max(axis=1)
    else:
        raise ValueError(f"unsupported safeguard source {config['source']}")
    support = (member_speed >= trigger[:, None]).sum(axis=1)
    base_speed = np.hypot(base_mu[:, 1], base_mu[:, 2])
    gate = (
        (expert_speed >= trigger)
        & (
            expert_speed
            > base_speed + float(config["minimum_speed_advantage_ms"])
        )
        & (support >= int(config["minimum_member_support"]))
    ).astype(np.float32)
    gate *= float(config["alpha"])
    output = base_mu.copy()
    output[:, 1] += gate * (expert_u - base_mu[:, 1])
    output[:, 2] += gate * (expert_v - base_mu[:, 2])
    return output.astype(np.float32), gate


def objective(
    crps_u: np.ndarray,
    crps_v: np.ndarray,
    crps_weight: np.ndarray,
    brier: np.ndarray,
    brier_weight: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    c_weight = np.maximum(crps_weight[dates].sum(), 1e-12)
    b_weight = np.maximum(brier_weight[dates].sum(), 1e-12)
    c_u = crps_u[:, dates].sum(axis=1) / c_weight
    c_v = crps_v[:, dates].sum(axis=1) / c_weight
    b = brier[:, dates].sum(axis=1) / b_weight
    base_index = int(np.where(np.isclose(SCALES, 1.0))[0][0])
    score = 0.5 * ((c_u + c_v) / max(c_u[base_index] + c_v[base_index], 1e-12))
    score += 0.5 * (b / max(b[base_index], 1e-12))
    return score, {"U10_CRPS": c_u, "V10_CRPS": c_v, "q975_Brier": b}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--safeguard-config", type=Path, default=WORK / "extreme_safeguard_2020.json"
    )
    parser.add_argument(
        "--output", type=Path, default=WORK / "focused_safeguard_scale_2020.json"
    )
    args = parser.parse_args()
    started = time.time()
    device = torch.device(f"cuda:{args.gpu}")

    development = json.loads(args.safeguard_config.read_text(encoding="utf-8"))
    config = development.get("selected_config")
    if not config:
        raise RuntimeError("the frozen 2020 safeguard configuration is unavailable")

    x = np.load(TC / "TEMPORAL_test_x.npy", mmap_mode="r")
    y = np.load(TC / "TEMPORAL_test_y.npy", mmap_mode="r")
    leads = np.load(TC / "TEMPORAL_test_lead.npy")
    coords = np.load(TC / "TEMPORAL_test_coords.npy")
    meta = np.load(CACHE / "sample_meta.npy")
    fields = meta[(meta[:, 0] >= 280) & (meta[:, 0] <= 350)]
    patch_day_all = np.repeat(fields[:, 0].astype(np.int64), 4)
    unique_days = np.unique(patch_day_all)
    day_map = {int(day): index for index, day in enumerate(unique_days)}
    day_ids_all = np.asarray([day_map[int(day)] for day in patch_day_all], np.int64)
    selected_indices = np.where(np.isin(leads, TARGET_LEAD_INDEX))[0]

    norm_mean = np.load(NORM / "norm_mean.npy").astype(np.float32)
    norm_std = np.load(NORM / "norm_std.npy").astype(np.float32)
    thresholds_all = load_thresholds()
    champion, champion_report = build(
        "base", "full", Path(CHECKPOINTS["champion"]), device
    )
    router_path = ROOT / "router_only_s123/best_v3_spf.pt"
    router, router_report = build("paefuse", "full", router_path, device)

    shape = (len(SCALES), len(unique_days))
    crps_u = np.zeros(shape, np.float64)
    crps_v = np.zeros(shape, np.float64)
    brier = np.zeros(shape, np.float64)
    crps_weight = np.zeros(len(unique_days), np.float64)
    brier_weight = np.zeros(len(unique_days), np.float64)

    scale_grid = torch.as_tensor(SCALES, dtype=torch.float32, device=device)[:, None, None, None]
    for start in range(0, len(selected_indices), args.batch_size):
        indices = selected_indices[start : start + args.batch_size]
        raw = np.stack(
            [np.asarray(x[i, :, CP : CP + P, CP : CP + P], np.float32) for i in indices]
        )
        truth = np.stack(
            [np.asarray(y[i, :, CP : CP + P, CP : CP + P], np.float32) for i in indices]
        )
        norm = (raw - norm_mean[None, :, None, None]) / norm_std[None, :, None, None]
        inputs = (
            torch.from_numpy(norm).to(device),
            torch.from_numpy(raw).to(device),
            torch.from_numpy(leads[indices].astype(np.int64)).to(device),
        )
        lat0 = torch.from_numpy((coords[indices, 0] + CP).astype(np.int64)).to(device)
        lon0 = torch.from_numpy((coords[indices, 1] + CP).astype(np.int64)).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            champion_mu, champion_sigma, _ = champion(*inputs, lat0=lat0, lon0=lon0)
            router_mu, router_sigma, _ = router(*inputs, lat0=lat0, lon0=lon0)
        base_mu, base_sigma = phase_selective_batch(
            champion_mu.float().cpu().numpy(),
            champion_sigma.float().cpu().numpy(),
            router_mu.float().cpu().numpy(),
            router_sigma.float().cpu().numpy(),
        )

        q95 = np.empty((len(indices), P, P), np.float32)
        q975 = np.empty_like(q95)
        for batch_i, index in enumerate(indices):
            lead = LEAD_HOURS[int(leads[index])]
            row = int(coords[index, 0]) + CP
            column = int(coords[index, 1]) + CP
            q95[batch_i] = thresholds_all["wind"][lead]["q95"][row : row + P, column : column + P]
            q975[batch_i] = thresholds_all["wind"][lead]["q975"][row : row + P, column : column + P]
        safeguarded_mu, gate = safeguard_batch(base_mu, raw, q95, config)

        mu = torch.as_tensor(safeguarded_mu[:, 1:3], dtype=torch.float32, device=device)
        sigma0 = torch.as_tensor(base_sigma[:, 1:3], dtype=torch.float32, device=device)
        truth_uv = torch.as_tensor(truth[:, 1:3], dtype=torch.float32, device=device)
        gate_t = torch.as_tensor(gate, dtype=torch.float32, device=device)
        sigma = sigma0[None] * (1.0 + gate_t[None, :, None] * (scale_grid[:, :, None] - 1.0))

        truth_speed = torch.hypot(truth_uv[:, 0], truth_uv[:, 1])
        observed = (truth_speed >= torch.as_tensor(q975, device=device)).float()
        focus = torch.maximum(gate_t, observed)
        weight = torch.as_tensor(latitude_weights(coords[indices]), dtype=torch.float32, device=device)
        focus_weight = weight * focus
        crps = normal_crps(mu[None], sigma, truth_uv[None])
        c_u = (crps[:, :, 0] * focus_weight[None]).sum(dim=(2, 3)).cpu().numpy()
        c_v = (crps[:, :, 1] * focus_weight[None]).sum(dim=(2, 3)).cpu().numpy()

        speed = torch.hypot(mu[:, 0], mu[:, 1])
        su, sv = sigma[:, :, 0].clamp_min(0.02), sigma[:, :, 1].clamp_min(0.02)
        speed_sd = torch.sqrt((mu[None, :, 0] * su).square() + (mu[None, :, 1] * sv).square())
        speed_sd = (speed_sd / speed[None].clamp_min(0.5)).clamp_min(0.05)
        z = (torch.as_tensor(q975, device=device)[None] - speed[None]) / (
            math.sqrt(2.0) * speed_sd
        )
        probability = (0.5 * torch.erfc(z)).clamp(0.0, 1.0)
        b = ((probability - observed[None]).square() * weight[None]).sum(dim=(2, 3)).cpu().numpy()

        day_ids = day_ids_all[indices]
        c_weights = focus_weight.sum(dim=(1, 2)).cpu().numpy()
        b_weights = weight.sum(dim=(1, 2)).cpu().numpy()
        for batch_i, day_id in enumerate(day_ids):
            crps_u[:, day_id] += c_u[:, batch_i]
            crps_v[:, day_id] += c_v[:, batch_i]
            brier[:, day_id] += b[:, batch_i]
            crps_weight[day_id] += c_weights[batch_i]
            brier_weight[day_id] += b_weights[batch_i]
        print(f"{min(start + len(indices), len(selected_indices))}/{len(selected_indices)}", flush=True)

    all_dates = np.arange(len(unique_days))
    folds = [part for part in np.array_split(all_dates, 4) if len(part)]
    cv: list[dict[str, object]] = []
    for fold_index, validation in enumerate(folds):
        training = np.setdiff1d(all_dates, validation)
        train_score, train_metrics = objective(
            crps_u, crps_v, crps_weight, brier, brier_weight, training
        )
        selected_index = int(np.argmin(train_score))
        validation_score, validation_metrics = objective(
            crps_u, crps_v, crps_weight, brier, brier_weight, validation
        )
        cv.append(
            {
                "fold": fold_index,
                "training_date_indices": training.tolist(),
                "validation_date_indices": validation.tolist(),
                "selected_scale": float(SCALES[selected_index]),
                "training_objective": float(train_score[selected_index]),
                "validation_objective": float(validation_score[selected_index]),
                "validation_metrics_selected": {
                    name: float(values[selected_index]) for name, values in validation_metrics.items()
                },
                "validation_metrics_scale_1": {
                    name: float(values[np.where(np.isclose(SCALES, 1.0))[0][0]])
                    for name, values in validation_metrics.items()
                },
            }
        )

    final_score, all_metrics = objective(
        crps_u, crps_v, crps_weight, brier, brier_weight, all_dates
    )
    selected_index = int(np.argmin(final_score))
    payload = {
        "status": "frozen",
        "year": 2020,
        "model_seed_used_for_shared_calibration": 123,
        "lead_hours": [72, 120, 168],
        "date_count": len(unique_days),
        "patch_count": len(selected_indices),
        "parameterization": "multiply U10/V10 sigma by one shared scalar only where the frozen forecast-only safeguard gate is active",
        "candidate_scales": SCALES.tolist(),
        "selection_objective": "equal-weighted relative mean U10/V10 CRPS on gate-or-observed-q97.5 pixels and global q97.5 Brier score",
        "cross_validation": "four contiguous initialization-date folds",
        "folds": cv,
        "final_scale": float(SCALES[selected_index]),
        "all_2020_objective_by_scale": final_score.tolist(),
        "all_2020_metrics_by_scale": {name: values.tolist() for name, values in all_metrics.items()},
        "information_contract": "no 2021/2022 observation, metric, date identity, or weather-type label was used",
        "safeguard_config": config,
        "inputs_sha256": {
            "safeguard_config": sha256(args.safeguard_config),
            "champion_checkpoint": sha256(Path(CHECKPOINTS["champion"])),
            "router_checkpoint": sha256(router_path),
        },
        "checkpoint_load_reports": {"champion": champion_report, "router": router_report},
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"final_scale": payload["final_scale"], "folds": cv}, indent=2), flush=True)
    print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
