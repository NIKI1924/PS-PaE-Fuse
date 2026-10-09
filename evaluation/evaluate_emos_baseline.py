#!/usr/bin/env python3
"""Matched heterogeneous EMOS baseline for the 2020 patches and 2021 fields.

The conditional mean is a lead-specific ridge-regularized affine combination
of the four heterogeneous members.  The conditional variance follows the
standard EMOS form c + d*S^2 and is fitted by Gaussian CRPS.  All parameters
are estimated from 2020 training patches only and then frozen for both the
2020 development period and the independent 2021 evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr


LEADS = (24, 48, 72, 96, 120, 144, 168)
TARGET_LEADS = (72, 120, 168)
LEAD_INDEX = {lead: index for index, lead in enumerate(LEADS)}
CP = 16
P = 96
VARIABLES = {"U10": 1, "V10": 2}


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, truth: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-4)
    z = (truth - mu) / sigma
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def collect_training(
    x: np.ndarray,
    y: np.ndarray,
    lead_index: np.ndarray,
    variable_index: int,
    rng: np.random.Generator,
    patches_per_lead: int,
    pixels_per_patch: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lead_i in range(len(LEADS)):
        candidates = np.flatnonzero(lead_index == lead_i)
        chosen = rng.choice(candidates, size=min(patches_per_lead, len(candidates)), replace=False)
        members, truth = [], []
        for sample_index in chosen:
            raw = np.asarray(x[sample_index, :, CP:CP + P, CP:CP + P], np.float32)
            target = np.asarray(y[sample_index, variable_index, CP:CP + P, CP:CP + P], np.float32)
            member_fields = raw.reshape(4, 6, P, P)[:, variable_index].reshape(4, -1).T
            pixels = rng.choice(P * P, size=min(pixels_per_patch, P * P), replace=False)
            members.append(member_fields[pixels])
            truth.append(target.reshape(-1)[pixels])
        output[lead_i] = (np.concatenate(members), np.concatenate(truth))
    return output


def fit_emos(members: np.ndarray, truth: np.ndarray, ridge: float) -> dict[str, object]:
    design = np.column_stack([np.ones(len(members), np.float64), members.astype(np.float64)])
    gram = design.T @ design
    penalty = ridge * np.trace(gram[1:, 1:]) / 4.0
    gram[1:, 1:] += np.eye(4) * penalty
    coefficients = np.linalg.solve(gram, design.T @ truth.astype(np.float64))
    mean = design @ coefficients
    residual = truth - mean
    spread = members.var(axis=1, ddof=1).astype(np.float64)
    initial_c = max(float(np.var(residual)) * 0.5, 1e-4)
    initial_d = max(float(np.var(residual)) * 0.5 / max(float(spread.mean()), 1e-4), 1e-4)

    def objective(log_parameters: np.ndarray) -> float:
        c, d = np.exp(log_parameters)
        sigma = np.sqrt(c + d * spread)
        return float(gaussian_crps(mean, sigma, truth).mean())

    optimized = minimize(
        objective,
        np.log([initial_c, initial_d]),
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0), (-12.0, 12.0)],
        options={"maxiter": 120},
    )
    c, d = np.exp(optimized.x)
    return {
        "mean_coefficients": coefficients.tolist(),
        "variance_c": float(c),
        "variance_d": float(d),
        "training_crps": float(optimized.fun),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "training_pixel_count": int(len(truth)),
    }


def apply_emos(members: np.ndarray, parameters: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(parameters["mean_coefficients"], np.float64)
    mean = coefficients[0] + np.tensordot(coefficients[1:], members, axes=(0, 0))
    spread = members.var(axis=0, ddof=1)
    sigma = np.sqrt(float(parameters["variance_c"]) + float(parameters["variance_d"]) * spread)
    return mean.astype(np.float32), sigma.astype(np.float32)


def interval(point: float, bootstrap: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(bootstrap, (2.5, 50.0, 97.5))
    return {
        "estimate": float(point),
        "bootstrap_median": float(median),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def summarize_2020(store: dict[str, np.ndarray], resamples: np.ndarray) -> dict[str, object]:
    weight = store["weight"]
    total_weight = weight.sum()
    boot_weight = weight[resamples].sum(axis=1)
    rmse = math.sqrt(store["se"].sum() / total_weight)
    rmse_boot = np.sqrt(store["se"][resamples].sum(axis=1) / boot_weight)
    crps = store["crps"].sum() / total_weight
    crps_boot = store["crps"][resamples].sum(axis=1) / boot_weight
    result: dict[str, object] = {"RMSE": interval(rmse, rmse_boot), "CRPS": interval(crps, crps_boot), "events": {}}
    for threshold in (8, 10, 12):
        counts = store[f"event_{threshold}"]
        point = counts.sum(axis=0)
        boot = counts[resamples].sum(axis=1)
        csi = point[0] / max(point.sum(), 1e-12)
        csi_boot = boot[:, 0] / np.maximum(boot.sum(axis=1), 1e-12)
        result["events"][str(threshold)] = {"CSI": interval(csi, csi_boot)}
    return result


def evaluate_2020(parameters: dict[str, object], bootstrap_count: int) -> dict[str, object]:
    root = Path("/vol2/xrx/temporal_cache")
    x = np.load(root / "TEMPORAL_test_x.npy", mmap_mode="r")
    y = np.load(root / "TEMPORAL_test_y.npy", mmap_mode="r")
    leads = np.load(root / "TEMPORAL_test_lead.npy")
    coords = np.load(root / "TEMPORAL_test_coords.npy")
    meta = np.load("/vol2/xrx/baseline_cache/sample_meta.npy")
    fields = meta[(meta[:, 0] >= 280) & (meta[:, 0] <= 350)]
    days = np.repeat(fields[:, 0].astype(np.int64), 4)
    unique_days = np.unique(days)
    day_map = {int(day): index for index, day in enumerate(unique_days)}
    day_ids = np.asarray([day_map[int(day)] for day in days])
    climatology = -0.014580273426514665
    store = {
        "se": np.zeros(len(unique_days), np.float64),
        "crps": np.zeros(len(unique_days), np.float64),
        "weight": np.zeros(len(unique_days), np.float64),
        **{f"event_{threshold}": np.zeros((len(unique_days), 3), np.float64) for threshold in (8, 10, 12)},
    }
    for sample_index in range(len(x)):
        lead_i = int(leads[sample_index])
        raw = np.asarray(x[sample_index, :, CP:CP + P, CP:CP + P], np.float32)
        members = raw.reshape(4, 6, P, P)[:, 1]
        truth = np.asarray(y[sample_index, 1, CP:CP + P, CP:CP + P], np.float32)
        mean, sigma = apply_emos(members, parameters["U10"][str(lead_i)])
        rows = coords[sample_index, 0] + CP + np.arange(P)
        latitude = 90.0 - 0.25 * rows
        weight = np.broadcast_to(np.maximum(np.cos(np.deg2rad(latitude)), 1e-3)[:, None], (P, P))
        date_index = day_ids[sample_index]
        store["se"][date_index] += float(((mean - truth) ** 2 * weight).sum())
        store["crps"][date_index] += float((gaussian_crps(mean, sigma, truth) * weight).sum())
        store["weight"][date_index] += float(weight.sum())
        for threshold in (8, 10, 12):
            forecast = np.abs(mean - climatology) >= threshold
            observed = np.abs(truth - climatology) >= threshold
            store[f"event_{threshold}"][date_index] += [
                float(weight[forecast & observed].sum()),
                float(weight[(~forecast) & observed].sum()),
                float(weight[forecast & (~observed)].sum()),
            ]
    rng = np.random.default_rng(20260905)
    resamples = rng.integers(0, len(unique_days), size=(bootstrap_count, len(unique_days)), dtype=np.int16)
    return {
        "date_count": len(unique_days),
        "patch_count": len(x),
        "metrics": summarize_2020(store, resamples),
    }


def summarize_2021_store(store: dict[str, object], resamples: np.ndarray) -> dict[str, object]:
    weight = store["weight"]
    wb = weight[resamples].sum(axis=1)
    result = {
        "U10_RMSE": interval(
            math.sqrt(store["se"].sum() / weight.sum()),
            np.sqrt(store["se"][resamples].sum(axis=1) / wb),
        ),
        "U10_CRPS": interval(
            store["crps"].sum() / weight.sum(),
            store["crps"][resamples].sum(axis=1) / wb,
        ),
        "events": {},
    }
    for event, counts in store["events"].items():
        point = counts.sum(axis=0)
        boot = counts[resamples].sum(axis=1)
        csi = point[0] / max(point.sum(), 1e-12)
        csi_boot = boot[:, 0] / np.maximum(boot.sum(axis=1), 1e-12)
        result["events"][event] = {"CSI": interval(csi, csi_boot)}
    return result


def evaluate_2021(parameters: dict[str, object], bootstrap_count: int) -> dict[str, object]:
    from evaluate_paefuse_crossyear_2021 import LONG_VARS, VARS, load_thresholds, sample_thresholds

    cache_root = Path("/vol2/xrx/baseline_cache_2021")
    hres_root = Path("/vol2/xrx/wb2_2021_unified/hres")
    caches = [np.load(cache_root / f"cache_{name}.npy", mmap_mode="r") for name in VARS]
    hres = [np.load(hres_root / f"{name}.npy", mmap_mode="r") for name in LONG_VARS]
    meta = np.load(cache_root / "sample_meta.npy")
    n_dates = int(meta[:, 0].max()) + 1
    latitude = np.linspace(90.0, -90.0, 721)
    weight = np.broadcast_to(np.maximum(np.cos(np.deg2rad(latitude)), 1e-3)[:, None], (721, 1440))
    thresholds_all = load_thresholds()
    stores = {
        str(lead): {
            "se": np.zeros(n_dates, np.float64),
            "crps": np.zeros(n_dates, np.float64),
            "weight": np.zeros(n_dates, np.float64),
            "events": {name: np.zeros((n_dates, 3), np.float64) for name in ("wind_q95", "wind_q975", "wind_abs15", "wind_abs20", "wind_abs25")},
        }
        for lead in TARGET_LEADS
    }
    for sample_index, row in enumerate(meta):
        date_index, old_hres_step, _, lead = map(int, row)
        if lead not in TARGET_LEADS:
            continue
        means, sigmas = {}, {}
        for variable, variable_index in VARIABLES.items():
            members = np.stack([
                np.asarray(caches[variable_index][sample_index, 0], np.float32),
                np.asarray(caches[variable_index][sample_index, 1], np.float32),
                np.asarray(caches[variable_index][sample_index, 2], np.float32),
                np.asarray(hres[variable_index][date_index, old_hres_step + 1, ::-1, :], np.float32),
            ])
            means[variable], sigmas[variable] = apply_emos(
                members, parameters[variable][str(LEAD_INDEX[lead])]
            )
        truth_u = np.asarray(caches[1][sample_index, 4], np.float32)
        truth_v = np.asarray(caches[2][sample_index, 4], np.float32)
        store = stores[str(lead)]
        store["se"][date_index] += float(((means["U10"] - truth_u) ** 2 * weight).sum())
        store["crps"][date_index] += float((gaussian_crps(means["U10"], sigmas["U10"], truth_u) * weight).sum())
        store["weight"][date_index] += float(weight.sum())
        forecast_speed = np.hypot(means["U10"], means["V10"])
        truth_speed = np.hypot(truth_u, truth_v)
        frozen = sample_thresholds(thresholds_all, lead)
        event_thresholds = {
            "wind_q95": frozen["wind_q95"],
            "wind_q975": frozen["wind_q975"],
            "wind_abs15": 15.0,
            "wind_abs20": 20.0,
            "wind_abs25": 25.0,
        }
        for event, threshold in event_thresholds.items():
            forecast = forecast_speed >= threshold
            observed = truth_speed >= threshold
            store["events"][event][date_index] += [
                float(weight[forecast & observed].sum()),
                float(weight[(~forecast) & observed].sum()),
                float(weight[forecast & (~observed)].sum()),
            ]
    rng = np.random.default_rng(20260905)
    resamples = rng.integers(0, n_dates, size=(bootstrap_count, n_dates), dtype=np.int16)
    return {
        "date_count": n_dates,
        "metrics": {lead: summarize_2021_store(store, resamples) for lead, store in stores.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches-per-lead", type=int, default=384)
    parser.add_argument("--pixels-per-patch", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xrx/wenqiong/paefuse_pr_20260904/emos_matched.json"),
    )
    args = parser.parse_args()
    started = time.time()
    root = Path("/vol2/xrx/temporal_cache")
    x = np.load(root / "TEMPORAL_train_x.npy", mmap_mode="r")
    y = np.load(root / "TEMPORAL_train_y.npy", mmap_mode="r")
    leads = np.load(root / "TEMPORAL_train_lead.npy")
    rng = np.random.default_rng(20260905)
    parameters: dict[str, object] = {}
    for variable, variable_index in VARIABLES.items():
        sampled = collect_training(
            x, y, leads, variable_index, rng,
            args.patches_per_lead, args.pixels_per_patch,
        )
        parameters[variable] = {
            str(lead_i): fit_emos(members, truth, args.ridge)
            for lead_i, (members, truth) in sampled.items()
        }
        print(f"fitted {variable}", flush=True)
    payload = {
        "status": "complete",
        "method": "heterogeneous Gaussian EMOS",
        "fit_contract": "lead-specific affine four-member mean fitted by ridge regression; variance c+d*S^2 fitted by Gaussian CRPS; 2020 training patches only",
        "seed": 20260905,
        "parameters": parameters,
        "validation_2020": evaluate_2020(parameters, args.bootstrap),
        "independent_2021": evaluate_2021(parameters, args.bootstrap),
        "bootstrap_replicates": args.bootstrap,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
