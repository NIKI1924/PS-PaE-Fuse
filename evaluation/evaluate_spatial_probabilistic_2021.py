#!/usr/bin/env python3
"""Scale-aware and probabilistic 2021 verification for phase-selective wind fusion.

This is a secondary, model-agnostic verification pass.  It evaluates the
already-frozen 2020 selector with (i) Fractions Skill Score (FSS) at several
spatial scales and (ii) Gaussian delta-method wind-speed exceedance Brier
scores and reliability counts.  Resampling is by complete initialization day.
No 2021 observation is used by the models, selector, thresholds, or case
selection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/xrx/wenqiong")
from evaluate_paefuse_crossyear_2021 import (
    LEAD_INDEX,
    LONG_VARS,
    TARGET_LEADS,
    VARS,
    build_model,
    infer_full,
    load_thresholds,
    normal_crps_fast,
    phase_selective_fusion,
    sample_thresholds,
    tile_starts,
)


ROOT = Path("/vol2/xrx/paefuse_pr_20260904")
MODEL_SPECS = {
    "champion": (
        "base",
        "full",
        Path("/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt"),
    ),
    "full_s123": ("paefuse", "full", ROOT / "full_s123/best_v3_spf.pt"),
    "full_s456": ("paefuse", "full", ROOT / "full_s456/best_v3_spf.pt"),
    "full_s789": ("paefuse", "full", ROOT / "full_s789/best_v3_spf.pt"),
    "router_s123": ("paefuse", "full", ROOT / "router_only_s123/best_v3_spf.pt"),
    "router_s456": ("paefuse", "full", ROOT / "router_only_s456/best_v3_spf.pt"),
    "router_s789": ("paefuse", "full", ROOT / "router_only_s789/best_v3_spf.pt"),
}
SELECTIVE = {
    "phase_selective_full_s123": "full_s123",
    "phase_selective_full_s456": "full_s456",
    "phase_selective_full_s789": "full_s789",
    "phase_selective_router_s123": "router_s123",
    "phase_selective_router_s456": "router_s456",
    "phase_selective_router_s789": "router_s789",
}
EVENTS = ("wind_q95", "wind_q975", "wind_abs15", "wind_abs20", "wind_abs25")
WINDOWS = (1, 5, 9, 17)
RELIABILITY_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def new_daily(n_dates: int) -> dict[str, object]:
    return {
        "u10_se": np.zeros(n_dates, np.float64),
        "u10_crps": np.zeros(n_dates, np.float64),
        "verification_weight": np.zeros(n_dates, np.float64),
        "event_counts": {
            event: np.zeros((n_dates, 3), np.float64) for event in EVENTS
        },
        "fss_num": {
            event: {str(window): np.zeros(n_dates, np.float64) for window in WINDOWS}
            for event in EVENTS
        },
        "fss_den": {
            event: {str(window): np.zeros(n_dates, np.float64) for window in WINDOWS}
            for event in EVENTS
        },
        "brier_sum": {event: np.zeros(n_dates, np.float64) for event in EVENTS},
        "brier_weight": {event: np.zeros(n_dates, np.float64) for event in EVENTS},
        "reliability": {
            event: {
                "weight": np.zeros(10, np.float64),
                "probability_sum": np.zeros(10, np.float64),
                "observed_sum": np.zeros(10, np.float64),
            }
            for event in EVENTS
        },
    }


def apply_emos_wind(
    raw: np.ndarray,
    parameters: dict[str, object],
    lead: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the 2020-fitted heterogeneous EMOS parameters to U10 and V10."""
    members = raw.reshape(4, 6, 721, 1440)
    mean = np.zeros((6, 721, 1440), np.float32)
    sigma = np.zeros_like(mean)
    for variable, variable_index in (("U10", 1), ("V10", 2)):
        fitted = parameters[variable][str(LEAD_INDEX[lead])]
        coefficients = np.asarray(fitted["mean_coefficients"], np.float64)
        member_fields = members[:, variable_index]
        mean[variable_index] = (
            coefficients[0]
            + np.tensordot(coefficients[1:], member_fields, axes=(0, 0))
        ).astype(np.float32)
        spread = member_fields.var(axis=0, ddof=1)
        sigma[variable_index] = np.sqrt(
            float(fitted["variance_c"]) + float(fitted["variance_d"]) * spread
        ).astype(np.float32)
    return mean, sigma


def event_thresholds(thresholds: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "wind_q95": torch.as_tensor(thresholds["wind_q95"], dtype=torch.float32, device=device),
        "wind_q975": torch.as_tensor(thresholds["wind_q975"], dtype=torch.float32, device=device),
        "wind_abs15": torch.tensor(15.0, dtype=torch.float32, device=device),
        "wind_abs20": torch.tensor(20.0, dtype=torch.float32, device=device),
        "wind_abs25": torch.tensor(25.0, dtype=torch.float32, device=device),
    }


def neighborhood_fraction(mask: torch.Tensor, window: int) -> torch.Tensor:
    if window == 1:
        return mask
    pad = window // 2
    field = F.pad(mask[None, None], (pad, pad, 0, 0), mode="circular")
    field = F.pad(field, (0, 0, pad, pad), mode="replicate")
    return F.avg_pool2d(field, kernel_size=window, stride=1)[0, 0]


def wind_probability(mu: torch.Tensor, sigma: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Delta-method Gaussian approximation for P(sqrt(U^2+V^2) >= threshold)."""
    u, v = mu[1], mu[2]
    su, sv = sigma[1].clamp_min(0.02), sigma[2].clamp_min(0.02)
    speed = torch.hypot(u, v)
    speed_sd = torch.sqrt((u * su).square() + (v * sv).square()) / speed.clamp_min(0.5)
    speed_sd = speed_sd.clamp_min(0.05)
    z = (threshold - speed) / (math.sqrt(2.0) * speed_sd)
    return (0.5 * torch.erfc(z)).clamp(0.0, 1.0)


def update_daily(
    store: dict[str, object],
    date_index: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    truth: np.ndarray,
    thresholds: dict[str, object],
    area_weight: torch.Tensor,
    device: torch.device,
) -> None:
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    truth_t = torch.as_tensor(truth, dtype=torch.float32, device=device)
    truth_speed = torch.hypot(truth_t[1], truth_t[2])
    forecast_speed = torch.hypot(mu_t[1], mu_t[2])
    store["u10_se"][date_index] += float(
        ((mu_t[1] - truth_t[1]).square() * area_weight).sum().item()
    )
    store["u10_crps"][date_index] += float(
        (torch.as_tensor(
            normal_crps_fast(mu[1], sigma[1], truth[1]),
            dtype=torch.float32,
            device=device,
        ) * area_weight).sum().item()
    )
    store["verification_weight"][date_index] += float(area_weight.sum().item())
    for event, threshold in event_thresholds(thresholds, device).items():
        observed = (truth_speed >= threshold).float()
        forecast = (forecast_speed >= threshold).float()
        counts = store["event_counts"][event]
        counts[date_index, 0] += float((area_weight * forecast * observed).sum().item())
        counts[date_index, 1] += float((area_weight * (1.0 - forecast) * observed).sum().item())
        counts[date_index, 2] += float((area_weight * forecast * (1.0 - observed)).sum().item())
        for window in WINDOWS:
            observed_fraction = neighborhood_fraction(observed, window)
            forecast_fraction = neighborhood_fraction(forecast, window)
            numerator = ((forecast_fraction - observed_fraction).square() * area_weight).sum()
            denominator = ((forecast_fraction.square() + observed_fraction.square()) * area_weight).sum()
            store["fss_num"][event][str(window)][date_index] += float(numerator.item())
            store["fss_den"][event][str(window)][date_index] += float(denominator.item())

        probability = wind_probability(mu_t, sigma_t, threshold)
        brier = (probability - observed).square()
        store["brier_sum"][event][date_index] += float((brier * area_weight).sum().item())
        store["brier_weight"][event][date_index] += float(area_weight.sum().item())

        bin_index = torch.clamp((probability * 10.0).long(), max=9).reshape(-1)
        weight_flat = area_weight.reshape(-1)
        probability_flat = probability.reshape(-1)
        observed_flat = observed.reshape(-1)
        rel = store["reliability"][event]
        rel["weight"] += torch.bincount(bin_index, weights=weight_flat, minlength=10).cpu().numpy()
        rel["probability_sum"] += torch.bincount(
            bin_index, weights=weight_flat * probability_flat, minlength=10
        ).cpu().numpy()
        rel["observed_sum"] += torch.bincount(
            bin_index, weights=weight_flat * observed_flat, minlength=10
        ).cpu().numpy()


def interval(point: float, bootstrap: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(bootstrap, (2.5, 50.0, 97.5))
    return {
        "estimate": float(point),
        "bootstrap_median": float(median),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize(store: dict[str, object], resamples: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    verification_weight = store["verification_weight"]
    bootstrap_weight = verification_weight[resamples].sum(axis=1)
    rmse = math.sqrt(store["u10_se"].sum() / max(verification_weight.sum(), 1e-12))
    rmse_boot = np.sqrt(
        store["u10_se"][resamples].sum(axis=1) / np.maximum(bootstrap_weight, 1e-12)
    )
    crps = store["u10_crps"].sum() / max(verification_weight.sum(), 1e-12)
    crps_boot = (
        store["u10_crps"][resamples].sum(axis=1) / np.maximum(bootstrap_weight, 1e-12)
    )
    result: dict[str, object] = {
        "traditional": {
            "U10_RMSE": interval(rmse, rmse_boot),
            "U10_CRPS": interval(crps, crps_boot),
            "events": {},
        },
        "FSS": {},
        "Brier": {},
        "reliability": {},
    }
    bootstrap_values: dict[str, np.ndarray] = {}
    bootstrap_values["traditional/U10_RMSE"] = rmse_boot
    bootstrap_values["traditional/U10_CRPS"] = crps_boot
    for event in EVENTS:
        counts = store["event_counts"][event]
        point_counts = counts.sum(axis=0)
        bootstrap_counts = counts[resamples].sum(axis=1)
        point_csi = point_counts[0] / max(point_counts.sum(), 1e-12)
        bootstrap_csi = bootstrap_counts[:, 0] / np.maximum(
            bootstrap_counts.sum(axis=1), 1e-12
        )
        point_total = verification_weight.sum()
        bootstrap_total = bootstrap_weight

        def contingency_scores(event_counts: np.ndarray, total_weight: np.ndarray) -> tuple[np.ndarray, ...]:
            hits, misses, false_alarms = (
                event_counts[..., 0], event_counts[..., 1], event_counts[..., 2]
            )
            correct_negatives = np.maximum(
                total_weight - hits - misses - false_alarms, 0.0
            )
            pod = hits / np.maximum(hits + misses, 1e-12)
            far = false_alarms / np.maximum(hits + false_alarms, 1e-12)
            frequency_bias = (hits + false_alarms) / np.maximum(hits + misses, 1e-12)
            false_alarm_rate = false_alarms / np.maximum(false_alarms + correct_negatives, 1e-12)
            eps = 1e-12
            hit_rate = np.clip(pod, eps, 1.0 - eps)
            false_alarm_rate = np.clip(false_alarm_rate, eps, 1.0 - eps)
            sedi = (
                np.log(false_alarm_rate) - np.log(hit_rate)
                - np.log1p(-false_alarm_rate) + np.log1p(-hit_rate)
            ) / (
                np.log(false_alarm_rate) + np.log(hit_rate)
                + np.log1p(-false_alarm_rate) + np.log1p(-hit_rate)
            )
            return pod, far, frequency_bias, sedi

        point_scores = contingency_scores(point_counts, np.asarray(point_total))
        bootstrap_scores = contingency_scores(bootstrap_counts, bootstrap_total)
        result["traditional"]["events"][event] = {
            "CSI": interval(point_csi, bootstrap_csi),
            "POD": interval(float(point_scores[0]), bootstrap_scores[0]),
            "FAR": interval(float(point_scores[1]), bootstrap_scores[1]),
            "FBIAS": interval(float(point_scores[2]), bootstrap_scores[2]),
            "SEDI": interval(float(point_scores[3]), bootstrap_scores[3]),
        }
        bootstrap_values[f"traditional/events/{event}/CSI"] = bootstrap_csi
        bootstrap_values[f"traditional/events/{event}/POD"] = bootstrap_scores[0]
        bootstrap_values[f"traditional/events/{event}/FAR"] = bootstrap_scores[1]
        bootstrap_values[f"traditional/events/{event}/FBIAS"] = bootstrap_scores[2]
        bootstrap_values[f"traditional/events/{event}/SEDI"] = bootstrap_scores[3]
        result["FSS"][event] = {}
        for window in WINDOWS:
            numerator = store["fss_num"][event][str(window)]
            denominator = store["fss_den"][event][str(window)]
            point = 1.0 - numerator.sum() / max(denominator.sum(), 1e-12)
            boot = 1.0 - numerator[resamples].sum(axis=1) / np.maximum(
                denominator[resamples].sum(axis=1), 1e-12
            )
            result["FSS"][event][str(window)] = interval(point, boot)
            bootstrap_values[f"FSS/{event}/{window}"] = boot

        total = store["brier_sum"][event]
        weight = store["brier_weight"][event]
        point = total.sum() / max(weight.sum(), 1e-12)
        boot = total[resamples].sum(axis=1) / np.maximum(weight[resamples].sum(axis=1), 1e-12)
        result["Brier"][event] = interval(point, boot)
        bootstrap_values[f"Brier/{event}"] = boot

        rel = store["reliability"][event]
        bin_weight = rel["weight"]
        result["reliability"][event] = {
            "bin_edges": RELIABILITY_EDGES.tolist(),
            "weight": bin_weight.tolist(),
            "mean_forecast_probability": np.divide(
                rel["probability_sum"], bin_weight,
                out=np.full(10, np.nan), where=bin_weight > 0,
            ).tolist(),
            "observed_frequency": np.divide(
                rel["observed_sum"], bin_weight,
                out=np.full(10, np.nan), where=bin_weight > 0,
            ).tolist(),
        }
    return result, bootstrap_values


def paired_intervals(
    bootstrap: dict[str, dict[str, dict[str, np.ndarray]]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for candidate in SELECTIVE:
        if candidate not in bootstrap:
            continue
        for reference in ("champion", "EMOS"):
            if reference not in bootstrap:
                continue
            output[candidate + "_vs_" + reference] = {}
            for lead in map(str, TARGET_LEADS):
                for metric, candidate_values in bootstrap[candidate][lead].items():
                    values = candidate_values - bootstrap[reference][lead][metric]
                    low, median, high = np.percentile(values, (2.5, 50.0, 97.5))
                    output[candidate + "_vs_" + reference][f"lead{lead}/{metric}"] = {
                        "median": float(median),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                    }
    for seed in (123, 456, 789):
        candidate = f"phase_selective_router_s{seed}"
        reference = f"phase_selective_full_s{seed}"
        if candidate not in bootstrap or reference not in bootstrap:
            continue
        key = candidate + "_vs_" + reference
        output[key] = {"isolated_factor": "remove_FFL_and_high_pass_objective"}
        for lead in map(str, TARGET_LEADS):
            for metric, candidate_values in bootstrap[candidate][lead].items():
                values = candidate_values - bootstrap[reference][lead][metric]
                low, median, high = np.percentile(values, (2.5, 50.0, 97.5))
                output[key][f"lead{lead}/{metric}"] = {
                    "median": float(median),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xrx/wenqiong/paefuse_pr_20260904/spatial_probabilistic_2021.json"),
    )
    args = parser.parse_args()
    started = time.time()
    device = torch.device(f"cuda:{args.gpu}")

    missing = {name: str(spec[2]) for name, spec in MODEL_SPECS.items() if not spec[2].exists()}
    if args.require_all and missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))
    specs = {name: spec for name, spec in MODEL_SPECS.items() if spec[2].exists()}
    active_selective = {
        name: phase_name for name, phase_name in SELECTIVE.items()
        if phase_name in specs and "champion" in specs
    }

    cache_root = Path("/vol2/xrx/baseline_cache_2021")
    hres_root = Path("/vol2/xrx/wb2_2021_unified/hres")
    caches = [np.load(cache_root / f"cache_{name}.npy", mmap_mode="r") for name in VARS]
    hres = [np.load(hres_root / f"{name}.npy", mmap_mode="r") for name in LONG_VARS]
    meta = np.load(cache_root / "sample_meta.npy")
    n_dates = int(meta[:, 0].max()) + 1
    norm_mean = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_mean.npy").astype(np.float32)
    norm_std = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_std.npy").astype(np.float32)
    thresholds_all = load_thresholds()
    emos_payload = json.loads(
        Path("/home/xrx/wenqiong/paefuse_pr_20260904/emos_matched.json").read_text(
            encoding="utf-8"
        )
    )
    emos_parameters = emos_payload["parameters"]
    selected = [index for index, row in enumerate(meta) if int(row[3]) in TARGET_LEADS]
    rows, columns = tile_starts(721), tile_starts(1440)
    area_weight = torch.as_tensor(
        np.broadcast_to(
            np.maximum(np.cos(np.deg2rad(np.linspace(90.0, -90.0, 721))), 1e-3)[:, None],
            (721, 1440),
        ).copy(),
        dtype=torch.float32,
        device=device,
    )

    stores = {
        method: {str(lead): new_daily(n_dates) for lead in TARGET_LEADS}
        for method in ("champion", "EMOS", *active_selective)
    }
    models = {}
    load_reports = {}
    for name, spec in specs.items():
        models[name], load_reports[name] = build_model(*spec, device)

    for count, sample_index in enumerate(selected, start=1):
        date_index, old_hres_step, _, lead = map(int, meta[sample_index])
        raw = np.empty((24, 721, 1440), np.float32)
        truth = np.empty((6, 721, 1440), np.float32)
        for variable_index in range(6):
            raw[variable_index] = np.asarray(caches[variable_index][sample_index, 0], np.float32)
            raw[6 + variable_index] = np.asarray(caches[variable_index][sample_index, 1], np.float32)
            raw[12 + variable_index] = np.asarray(caches[variable_index][sample_index, 2], np.float32)
            raw[18 + variable_index] = np.asarray(
                hres[variable_index][date_index, old_hres_step + 1, ::-1, :], np.float32
            )
            truth[variable_index] = np.asarray(caches[variable_index][sample_index, 4], np.float32)
        norm = (raw - norm_mean[:, None, None]) / norm_std[:, None, None]
        learned_mu: dict[str, np.ndarray] = {}
        learned_sigma: dict[str, np.ndarray] = {}
        for name, model in models.items():
            mu, sigma = infer_full(
                model, raw, norm, LEAD_INDEX[lead], rows, columns, args.batch_size, device
            )
            learned_mu[name] = mu
            learned_sigma[name] = sigma
        thresholds = sample_thresholds(thresholds_all, lead)
        emos_mu, emos_sigma = apply_emos_wind(raw, emos_parameters, lead)
        update_daily(
            stores["EMOS"][str(lead)], date_index,
            emos_mu, emos_sigma, truth,
            thresholds, area_weight, device,
        )
        update_daily(
            stores["champion"][str(lead)], date_index,
            learned_mu["champion"], learned_sigma["champion"], truth,
            thresholds, area_weight, device,
        )
        for method, phase_name in active_selective.items():
            mu, sigma, _ = phase_selective_fusion(
                learned_mu["champion"], learned_sigma["champion"],
                learned_mu[phase_name], learned_sigma[phase_name],
            )
            update_daily(
                stores[method][str(lead)], date_index, mu, sigma, truth,
                thresholds, area_weight, device,
            )
        print(f"{count}/{len(selected)} date={date_index} lead={lead}", flush=True)

    rng = np.random.default_rng(20260905)
    resamples = rng.integers(0, n_dates, size=(args.bootstrap, n_dates), dtype=np.int16)
    metrics: dict[str, object] = {}
    bootstrap: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for method, by_lead in stores.items():
        metrics[method] = {}
        bootstrap[method] = {}
        for lead, store in by_lead.items():
            metrics[method][lead], bootstrap[method][lead] = summarize(store, resamples)

    payload = {
        "status": "complete",
        "year": 2021,
        "scope": "secondary scale-aware and Gaussian delta-method probabilistic verification",
        "date_count": n_dates,
        "lead_hours": list(TARGET_LEADS),
        "events": list(EVENTS),
        "fss_windows_grid_cells": list(WINDOWS),
        "grid_spacing_degrees": 0.25,
        "fss_boundary_contract": "periodic longitude, replicated latitude; cosine-latitude weighting",
        "probability_contract": "independent-component Gaussian delta approximation from fused U/V means and scales; no post-hoc calibration",
        "emos_contract": emos_payload["fit_contract"],
        "resampling_unit": "initialization date",
        "bootstrap_replicates": args.bootstrap,
        "load_reports": load_reports,
        "missing_checkpoints": missing,
        "metrics": metrics,
        "paired_deltas": paired_intervals(bootstrap),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
