#!/usr/bin/env python3
"""Independent 2021 full-field validation for PaE-Fuse.

The evaluation uses the corrected exact-lead HRES contract, ERA5 truth, a
date-block bootstrap, thresholds fitted outside 2021, and case selection based
only on the verifying truth.  The default comparison contains the established
champion, the legacy phase-aware-loss model, and three independently trained
PaE-Fuse seeds.  Architecture ablations can be included after their checkpoints
finish by passing --include-ablations.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/xrx/wenqiong")
from paefuse import PaEFuseNet
from unified_phase_scorecard import normal_crps_fast
from wenqiong_v3_spf_dem import WenqiongV3SPFDEM


VARS = ("T2M", "U10", "V10", "MSL", "Z500", "T850")
LONG_VARS = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_500",
    "temperature_850",
)
TARGET_LEADS = (72, 120, 168)
LEAD_INDEX = {72: 2, 120: 4, 168: 6}
BASELINES = ("Pangu", "GraphCast", "FuXi", "HRES_corrected", "SimpleAvg")
SELECTIVE_MODELS = {
    "phase_selective_s123": "paefuse_full_s123",
    "phase_selective_s456": "paefuse_full_s456",
    "phase_selective_s789": "paefuse_full_s789",
    "selective_amplitude_s123": "paefuse_amplitude_s123",
    "selective_no_extreme_s123": "paefuse_no_extreme_s123",
    "selective_router_only_s123": "paefuse_router_only_s123",
    "selective_loss_only_s123": "phase_loss_only_s123",
}
MODEL_SPECS = {
    "champion": ("base", "full", Path("/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt")),
    "legacy_phase_loss": ("base", "full", Path("/home/xrx/wenqiong/ov_ffl2_s123/best_v3_spf.pt")),
    "paefuse_full_s123": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/full_s123/best_v3_spf.pt")),
    "paefuse_full_s456": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/full_s456/best_v3_spf.pt")),
    "paefuse_full_s789": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/full_s789/best_v3_spf.pt")),
}
ABLATION_SPECS = {
    "paefuse_amplitude_s123": ("paefuse", "amplitude_only", Path("/vol2/xrx/paefuse_pr_20260904/amplitude_s123/best_v3_spf.pt")),
    "paefuse_no_extreme_s123": ("paefuse", "no_extreme_condition", Path("/vol2/xrx/paefuse_pr_20260904/no_extreme_s123/best_v3_spf.pt")),
    "paefuse_router_only_s123": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/router_only_s123/best_v3_spf.pt")),
    "phase_loss_only_s123": ("base", "full", Path("/vol2/xrx/paefuse_pr_20260904/loss_only_s123/best_v3_spf.pt")),
}
ROUTER_MULTISEED_SPECS = {
    "paefuse_router_only_s456": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/router_only_s456/best_v3_spf.pt")),
    "paefuse_router_only_s789": ("paefuse", "full", Path("/vol2/xrx/paefuse_pr_20260904/router_only_s789/best_v3_spf.pt")),
}
ROUTER_MULTISEED_SELECTIVE = {
    "selective_router_only_s456": "paefuse_router_only_s456",
    "selective_router_only_s789": "paefuse_router_only_s789",
}
PH = PW = 96


def tile_starts(length: int) -> list[int]:
    starts = list(range(0, length - PH + 1, PH))
    if starts[-1] != length - PH:
        starts.append(length - PH)
    return starts


def event_names() -> tuple[str, ...]:
    tails = tuple(f"{var}_{tail}" for var in VARS for tail in ("p05", "p95"))
    return ("wind_q95", "wind_q975", "wind_abs15", "wind_abs20", "wind_abs25", *tails)


def new_daily(n_dates: int) -> dict[str, object]:
    return {
        "se": np.zeros((n_dates, len(VARS)), np.float64),
        "crps": np.zeros((n_dates, len(VARS)), np.float64),
        "weight": np.zeros(n_dates, np.float64),
        "events": {
            name: np.zeros((n_dates, 3), np.float64) for name in event_names()
        },
    }


def event_fields(
    prediction: np.ndarray,
    truth: np.ndarray,
    thresholds: dict[str, object],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | float]]:
    pred_speed = np.hypot(prediction[1], prediction[2])
    truth_speed = np.hypot(truth[1], truth[2])
    specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | float]] = {
        "wind_q95": (pred_speed, truth_speed, thresholds["wind_q95"]),
        "wind_q975": (pred_speed, truth_speed, thresholds["wind_q975"]),
        "wind_abs15": (pred_speed, truth_speed, 15.0),
        "wind_abs20": (pred_speed, truth_speed, 20.0),
        "wind_abs25": (pred_speed, truth_speed, 25.0),
    }
    tails = thresholds["tails"]
    for vi, var in enumerate(VARS):
        specs[f"{var}_p05"] = (-prediction[vi], -truth[vi], -tails[var]["p05"])
        specs[f"{var}_p95"] = (prediction[vi], truth[vi], tails[var]["p95"])
    return specs


def add_field(
    store: dict[str, object],
    date_i: int,
    prediction: np.ndarray,
    truth: np.ndarray,
    sigma: np.ndarray | None,
    weight: np.ndarray,
    thresholds: dict[str, object],
) -> None:
    error = prediction - truth
    store["se"][date_i] += (error * error * weight[None]).sum(axis=(1, 2))
    score = np.abs(error) if sigma is None else normal_crps_fast(prediction, sigma, truth)
    store["crps"][date_i] += (score * weight[None]).sum(axis=(1, 2))
    store["weight"][date_i] += weight.sum()
    for name, (pred, observed, threshold) in event_fields(prediction, truth, thresholds).items():
        p = pred >= threshold
        t = observed >= threshold
        store["events"][name][date_i, 0] += float((weight * (p & t)).sum())
        store["events"][name][date_i, 1] += float((weight * ((~p) & t)).sum())
        store["events"][name][date_i, 2] += float((weight * (p & (~t))).sum())


def interval(point: float, values: np.ndarray) -> dict[str, float]:
    lo, median, hi = np.percentile(values, (2.5, 50.0, 97.5))
    return {
        "estimate": float(point),
        "bootstrap_median": float(median),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def summarize(
    store: dict[str, object],
    resamples: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    w0 = store["weight"].sum()
    wb = store["weight"][resamples].sum(axis=1)
    result: dict[str, object] = {"variables": {}, "events": {}}
    boot_flat: dict[str, np.ndarray] = {}
    for vi, name in enumerate(VARS):
        se0 = store["se"][:, vi].sum()
        seb = store["se"][resamples, vi].sum(axis=1)
        c0 = store["crps"][:, vi].sum()
        cb = store["crps"][resamples, vi].sum(axis=1)
        rmse_b = np.sqrt(seb / wb)
        crps_b = cb / wb
        result["variables"][name] = {
            "RMSE": interval(np.sqrt(se0 / w0), rmse_b),
            "CRPS": interval(c0 / w0, crps_b),
        }
        boot_flat[f"variables/{name}/RMSE"] = rmse_b
        boot_flat[f"variables/{name}/CRPS"] = crps_b
    for name, daily in store["events"].items():
        point = daily.sum(axis=0)
        boot = daily[resamples].sum(axis=1)

        def compute(x: np.ndarray) -> tuple[np.ndarray, ...]:
            h, m, f = x[..., 0], x[..., 1], x[..., 2]
            return (
                h / np.maximum(h + m + f, 1e-12),
                h / np.maximum(h + m, 1e-12),
                f / np.maximum(h + f, 1e-12),
                (h + f) / np.maximum(h + m, 1e-12),
            )

        point_values = compute(point)
        boot_values = compute(boot)
        result["events"][name] = {}
        for metric_i, metric in enumerate(("CSI", "POD", "FAR", "FBIAS")):
            result["events"][name][metric] = interval(
                point_values[metric_i], boot_values[metric_i]
            )
            boot_flat[f"events/{name}/{metric}"] = boot_values[metric_i]
    return result, boot_flat


def build_model(kind: str, variant: str, checkpoint: Path, device: torch.device):
    kwargs = dict(
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
    model = PaEFuseNet(variant=variant, **kwargs) if kind == "paefuse" else WenqiongV3SPFDEM(**kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint),
        "kind": kind,
        "variant": variant,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def infer_full(
    model: torch.nn.Module,
    raw: np.ndarray,
    norm: np.ndarray,
    lead_idx: int,
    rows: list[int],
    columns: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tiles = [(row, column) for row in rows for column in columns]
    mu_sum = np.zeros((6, 721, 1440), np.float32)
    sigma_sum = np.zeros((6, 721, 1440), np.float32)
    count = np.zeros((721, 1440), np.float32)
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        batch_norm = np.stack([norm[:, r:r + PH, c:c + PW] for r, c in batch])
        batch_raw = np.stack([raw[:, r:r + PH, c:c + PW] for r, c in batch])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            mu, sigma, _ = model(
                torch.from_numpy(batch_norm).to(device),
                torch.from_numpy(batch_raw).to(device),
                torch.full((len(batch),), lead_idx, dtype=torch.long, device=device),
                lat0=torch.tensor([r for r, _ in batch], dtype=torch.long, device=device),
                lon0=torch.tensor([c for _, c in batch], dtype=torch.long, device=device),
            )
        mu = mu.float().cpu().numpy()
        sigma = sigma.float().cpu().numpy()
        for batch_i, (row, column) in enumerate(batch):
            mu_sum[:, row:row + PH, column:column + PW] += mu[batch_i]
            sigma_sum[:, row:row + PH, column:column + PW] += sigma[batch_i]
            count[row:row + PH, column:column + PW] += 1
    return mu_sum / count[None], sigma_sum / count[None]


def load_thresholds() -> dict[str, object]:
    threshold_dir = Path("/home/xrx/wenqiong/eval_results/thresholds")
    tails = {
        var: {
            "p05": np.asarray(np.load(threshold_dir / f"{var}_p5.npy"), np.float32),
            "p95": np.asarray(np.load(threshold_dir / f"{var}_p95.npy"), np.float32),
        }
        for var in VARS
    }
    with np.load("/vol2/xrx/extreme_spatial_demo_2020/thresholds_q95_q975.npz") as archive:
        wind = {
            lead: {
                "q95": np.asarray(archive[f"q95_{lead}"], np.float32),
                "q975": np.asarray(archive[f"q975_{lead}"], np.float32),
            }
            for lead in TARGET_LEADS
        }
    return {"tails": tails, "wind": wind}


def sample_thresholds(all_thresholds: dict[str, object], lead: int) -> dict[str, object]:
    return {
        "wind_q95": all_thresholds["wind"][lead]["q95"],
        "wind_q975": all_thresholds["wind"][lead]["q975"],
        "tails": all_thresholds["tails"],
    }


def select_cases(
    selected_indices: list[int],
    meta: np.ndarray,
    caches: list[np.ndarray],
    all_thresholds: dict[str, object],
    weight: np.ndarray,
) -> dict[str, dict[str, object]]:
    """Truth-only case selection, fixed before any model inference."""
    best = {
        "wind_q975_lead72": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "required_lead": 72},
        "wind_q975_lead120": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "required_lead": 120},
        "wind_q975_lead168": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "required_lead": 168},
        "wind_abs15": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "absolute_threshold": 15.0},
        "wind_abs20": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "absolute_threshold": 20.0},
        "wind_abs25": {"score": -1.0, "sample_index": -1, "variable": "wind_speed", "absolute_threshold": 25.0},
        "T2M_p95": {"score": -1.0, "sample_index": -1, "variable": "T2M"},
        "T2M_p05": {"score": -1.0, "sample_index": -1, "variable": "T2M"},
        "MSL_p05": {"score": -1.0, "sample_index": -1, "variable": "MSL"},
        "Z500_p95": {"score": -1.0, "sample_index": -1, "variable": "Z500"},
    }
    variable_index = {name: index for index, name in enumerate(VARS)}
    for sample_index in selected_indices:
        date_i, _, _, lead = map(int, meta[sample_index])
        truth = np.stack([np.asarray(cache[sample_index, 4], np.float32) for cache in caches])
        thresholds = sample_thresholds(all_thresholds, lead)
        speed = np.hypot(truth[1], truth[2])
        masks = {
            f"wind_q975_lead{lead}": speed >= thresholds["wind_q975"],
            "wind_abs15": speed >= 15.0,
            "wind_abs20": speed >= 20.0,
            "wind_abs25": speed >= 25.0,
            "T2M_p95": truth[0] >= thresholds["tails"]["T2M"]["p95"],
            "T2M_p05": truth[0] <= thresholds["tails"]["T2M"]["p05"],
            "MSL_p05": truth[3] <= thresholds["tails"]["MSL"]["p05"],
            "Z500_p95": truth[4] >= thresholds["tails"]["Z500"]["p95"],
        }
        for event, mask in masks.items():
            if event not in best:
                continue
            score = float((weight * mask).sum())
            if score > best[event]["score"]:
                best[event].update({"score": score, "sample_index": sample_index, "date_index": date_i, "lead_hours": lead})
    return best


def extract_case_field(values: np.ndarray, variable: str) -> np.ndarray:
    if variable == "wind_speed":
        return np.hypot(values[1], values[2]).astype(np.float32)
    # Copy the selected plane so retained case arrays do not keep the complete
    # six-variable prediction cube alive for the remainder of evaluation.
    return np.asarray(values[VARS.index(variable)], np.float32).copy()


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def phase_selective_fusion(
    champion_mu: np.ndarray,
    champion_sigma: np.ndarray,
    phase_mu: np.ndarray,
    phase_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen gate transferred from 2020 without 2021 parameter fitting.

    For U10 this is exactly the development-selected rule
    sigmoid(1.5*(max anomaly-8))*sigmoid(10*(phase anomaly-champion anomaly)).
    The same dimensionless rule is applied to the other variables using the
    pre-existing L3 training-event scales, with no cross-year tuning.
    """
    climatology = np.asarray(
        [284.86, 0.04, 0.19, 101051.80, 55100.46, 279.33], np.float32
    )[:, None, None]
    scale = np.asarray([8.0, 8.0, 7.0, 1200.0, 3000.0, 7.0], np.float32)[:, None, None]
    champion_anomaly = np.abs(champion_mu - climatology)
    phase_anomaly = np.abs(phase_mu - climatology)
    maximum_normalized_anomaly = np.maximum(champion_anomaly, phase_anomaly) / scale
    normalized_advantage = (phase_anomaly - champion_anomaly) / scale
    event_gate = stable_sigmoid(12.0 * (maximum_normalized_anomaly - 1.0))
    advantage_gate = stable_sigmoid(80.0 * normalized_advantage)
    gate = event_gate * advantage_gate
    mu = champion_mu + gate * (phase_mu - champion_mu)
    sigma = champion_sigma + gate * (phase_sigma - champion_sigma)
    return mu.astype(np.float32), sigma.astype(np.float32), gate.astype(np.float32)


def save_case(
    cases_dir: Path,
    case_name: str,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> str:
    path = cases_dir / f"{case_name}.npz"
    payload = {name: np.asarray(value, np.float32) for name, value in arrays.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    return str(path)


def paired_summary(
    candidates: dict[str, dict[str, np.ndarray]],
    references: tuple[str, ...] = ("champion", "legacy_phase_loss"),
) -> dict[str, object]:
    output: dict[str, object] = {}
    for candidate, candidate_values in candidates.items():
        if not candidate.startswith(("paefuse_full", "phase_selective", "selective_")):
            continue
        for reference in references:
            if reference not in candidates:
                continue
            key = f"{candidate}_vs_{reference}"
            output[key] = {}
            for metric, values in candidate_values.items():
                if metric not in candidates[reference]:
                    continue
                delta = values - candidates[reference][metric]
                low, median, high = np.percentile(delta, (2.5, 50.0, 97.5))
                output[key][metric] = {
                    "median": float(median),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }
    return output


def add_controlled_router_pairs(
    output: dict[str, object],
    candidates: dict[str, dict[str, np.ndarray]],
) -> None:
    """Add seed-matched router-only minus phase-objective comparisons."""
    for seed in (123, 456, 789):
        candidate = f"selective_router_only_s{seed}"
        reference = f"phase_selective_s{seed}"
        if candidate not in candidates or reference not in candidates:
            continue
        key = f"{candidate}_vs_{reference}"
        output[key] = {"isolated_factor": "remove_FFL_and_high_pass_objective"}
        for metric, values in candidates[candidate].items():
            if metric not in candidates[reference]:
                continue
            delta = values - candidates[reference][metric]
            low, median, high = np.percentile(delta, (2.5, 50.0, 97.5))
            output[key][metric] = {
                "median": float(median),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }


def seed_summary(metrics: dict[str, object], prefix: str = "phase_selective_s") -> dict[str, object]:
    names = [name for name in metrics if name.startswith(prefix)]
    output: dict[str, object] = {"seed_models": names, "variables": {}, "events": {}}
    if not names:
        return output
    for var in VARS:
        output["variables"][var] = {}
        for metric in ("RMSE", "CRPS"):
            values = np.asarray([metrics[name]["variables"][var][metric]["estimate"] for name in names])
            output["variables"][var][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    for event in event_names():
        values = np.asarray([metrics[name]["events"][event]["CSI"]["estimate"] for name in names])
        output["events"][event] = {
            "CSI_mean": float(values.mean()),
            "CSI_sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "CSI_min": float(values.min()),
            "CSI_max": float(values.max()),
        }
    return output


def new_gate_stats() -> dict[str, object]:
    return {
        "sum_per_variable": np.zeros(len(VARS), np.float64),
        "pixel_count": 0,
        "wind_extreme_sum": 0.0,
        "wind_extreme_count": 0,
        "wind_nonextreme_sum": 0.0,
        "wind_nonextreme_count": 0,
    }


def finalize_gate_stats(store: dict[str, object]) -> dict[str, object]:
    return {
        "mean_gate_per_variable": (
            store["sum_per_variable"] / max(int(store["pixel_count"]), 1)
        ).tolist(),
        "mean_uv_gate_on_truth_wind_q975": float(
            store["wind_extreme_sum"] / max(int(store["wind_extreme_count"]), 1)
        ),
        "mean_uv_gate_elsewhere": float(
            store["wind_nonextreme_sum"] / max(int(store["wind_nonextreme_count"]), 1)
        ),
        "truth_wind_q975_pixel_count": int(store["wind_extreme_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument(
        "--include-router-multiseed",
        action="store_true",
        help="also evaluate the phase-router-only checkpoints for seeds 456 and 789",
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xrx/wenqiong/paefuse_pr_20260904/crossyear_2021.json"),
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("/vol2/xrx/paefuse_pr_20260904/cases_2021"),
    )
    args = parser.parse_args()
    started = time.time()
    device = torch.device(f"cuda:{args.gpu}")

    cache_root = Path("/vol2/xrx/baseline_cache_2021")
    hres_root = Path("/vol2/xrx/wb2_2021_unified/hres")
    caches = [np.load(cache_root / f"cache_{name}.npy", mmap_mode="r") for name in VARS]
    hres = [np.load(hres_root / f"{name}.npy", mmap_mode="r") for name in LONG_VARS]
    meta = np.load(cache_root / "sample_meta.npy")
    n_dates = int(meta[:, 0].max()) + 1
    tags = sorted(
        re.search(r"_(\d{10})\.npz", path).group(1)
        for path in glob.glob("/vol2/xrx/wb2_2021/graphcast_npz/gc_*.npz")
    )[:n_dates]
    norm_mean = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_mean.npy").astype(np.float32)
    norm_std = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_std.npy").astype(np.float32)
    thresholds_all = load_thresholds()

    latitude = np.linspace(90.0, -90.0, 721)
    longitude = np.linspace(0.0, 359.75, 1440)
    weight = np.broadcast_to(
        np.maximum(np.cos(np.deg2rad(latitude)), 1e-3)[:, None], (721, 1440)
    )
    rows, columns = tile_starts(721), tile_starts(1440)
    selected = [index for index, row in enumerate(meta) if int(row[3]) in TARGET_LEADS]
    selected_cases = select_cases(selected, meta, caches, thresholds_all, weight)
    selected_case_by_sample: dict[int, list[str]] = {}
    for case_name, information in selected_cases.items():
        selected_case_by_sample.setdefault(int(information["sample_index"]), []).append(case_name)

    specs = dict(MODEL_SPECS)
    if args.include_ablations:
        specs.update(ABLATION_SPECS)
    if args.include_router_multiseed:
        specs.update(ROUTER_MULTISEED_SPECS)
    missing_checkpoints = {name: str(spec[2]) for name, spec in specs.items() if not spec[2].exists()}
    if args.require_all and missing_checkpoints:
        raise FileNotFoundError(json.dumps(missing_checkpoints, indent=2))
    specs = {name: spec for name, spec in specs.items() if spec[2].exists()}
    requested_selective_models = dict(SELECTIVE_MODELS)
    if args.include_router_multiseed:
        requested_selective_models.update(ROUTER_MULTISEED_SELECTIVE)
    active_selective_models = {
        name: phase_name
        for name, phase_name in requested_selective_models.items()
        if phase_name in specs and "champion" in specs
    }

    methods = {
        name: {lead: new_daily(n_dates) for lead in TARGET_LEADS}
        for name in (*BASELINES, *specs, *active_selective_models)
    }
    model_objects = {}
    load_reports = {}
    selective_gate_stats = {
        name: {lead: new_gate_stats() for lead in TARGET_LEADS}
        for name in active_selective_models
    }
    for name, (kind, variant, checkpoint) in specs.items():
        model_objects[name], load_reports[name] = build_model(kind, variant, checkpoint, device)

    args.cases_dir.mkdir(parents=True, exist_ok=True)
    case_arrays: dict[str, dict[str, np.ndarray]] = {name: {} for name in selected_cases}
    for count_i, sample_i in enumerate(selected):
        date_i, old_hres_step, _, lead = map(int, meta[sample_i])
        raw = np.empty((24, 721, 1440), np.float32)
        truth = np.empty((6, 721, 1440), np.float32)
        for vi in range(6):
            raw[vi] = np.asarray(caches[vi][sample_i, 0], np.float32)
            raw[6 + vi] = np.asarray(caches[vi][sample_i, 1], np.float32)
            raw[12 + vi] = np.asarray(caches[vi][sample_i, 2], np.float32)
            raw[18 + vi] = np.asarray(hres[vi][date_i, old_hres_step + 1, ::-1, :], np.float32)
            truth[vi] = np.asarray(caches[vi][sample_i, 4], np.float32)
        thresholds = sample_thresholds(thresholds_all, lead)
        members = raw.reshape(4, 6, 721, 1440)
        predictions = {
            BASELINES[index]: members[index] for index in range(4)
        }
        predictions["SimpleAvg"] = members.mean(axis=0)
        for name, prediction in predictions.items():
            add_field(methods[name][lead], date_i, prediction, truth, None, weight, thresholds)

        norm = (raw - norm_mean[:, None, None]) / norm_std[:, None, None]
        learned_predictions: dict[str, np.ndarray] = {}
        learned_sigmas: dict[str, np.ndarray] = {}
        for name, model in model_objects.items():
            mu, sigma = infer_full(
                model, raw, norm, LEAD_INDEX[lead], rows, columns, args.batch_size, device
            )
            learned_predictions[name] = mu
            learned_sigmas[name] = sigma
            add_field(methods[name][lead], date_i, mu, truth, sigma, weight, thresholds)

        selective_predictions = {}
        selective_gates = {}
        for selective_name, phase_name in active_selective_models.items():
            mu, sigma, gate = phase_selective_fusion(
                learned_predictions["champion"],
                learned_sigmas["champion"],
                learned_predictions[phase_name],
                learned_sigmas[phase_name],
            )
            selective_predictions[selective_name] = mu
            selective_gates[selective_name] = gate
            add_field(
                methods[selective_name][lead], date_i, mu, truth, sigma, weight, thresholds
            )
            gate_store = selective_gate_stats[selective_name][lead]
            gate_store["sum_per_variable"] += gate.sum(axis=(1, 2))
            gate_store["pixel_count"] += gate.shape[1] * gate.shape[2]
            uv_gate = 0.5 * (gate[1] + gate[2])
            truth_wind = np.hypot(truth[1], truth[2])
            extreme_mask = truth_wind >= thresholds["wind_q975"]
            gate_store["wind_extreme_sum"] += float(uv_gate[extreme_mask].sum())
            gate_store["wind_extreme_count"] += int(extreme_mask.sum())
            gate_store["wind_nonextreme_sum"] += float(uv_gate[~extreme_mask].sum())
            gate_store["wind_nonextreme_count"] += int((~extreme_mask).sum())

        if sample_i in selected_case_by_sample:
            for case_name in selected_case_by_sample[sample_i]:
                variable = str(selected_cases[case_name]["variable"])
                arrays = case_arrays[case_name]
                arrays["truth"] = extract_case_field(truth, variable)
                for name, prediction in {
                    **predictions,
                    **learned_predictions,
                    **selective_predictions,
                }.items():
                    arrays[name] = extract_case_field(prediction, variable)
                if case_name.startswith("wind_q975"):
                    arrays["threshold"] = thresholds["wind_q975"]
                elif case_name.startswith("wind_abs"):
                    value = float(case_name.removeprefix("wind_abs"))
                    arrays["threshold"] = np.full_like(arrays["truth"], value)
                else:
                    var, tail = case_name.split("_", 1)
                    arrays["threshold"] = thresholds["tails"][var][tail]

        print(
            f"{count_i + 1}/{len(selected)} date={date_i} "
            f"tag={tags[date_i] if date_i < len(tags) else date_i} lead={lead}",
            flush=True,
        )

    rng = np.random.default_rng(20260904)
    resamples = rng.integers(0, n_dates, size=(args.bootstrap, n_dates), dtype=np.int16)
    metrics: dict[str, object] = {}
    bootstrap_values: dict[str, dict[str, np.ndarray]] = {}
    for name, by_lead in methods.items():
        metrics[name] = {}
        bootstrap_values[name] = {}
        for lead, store in by_lead.items():
            summary, boot = summarize(store, resamples)
            metrics[name][str(lead)] = summary
            bootstrap_values[name].update({f"lead{lead}/{key}": value for key, value in boot.items()})

    case_manifest = {}
    for case_name, information in selected_cases.items():
        sample_i = int(information["sample_index"])
        date_i = int(information["date_index"])
        metadata = {
            **information,
            "case_name": case_name,
            "initialization_tag": tags[date_i] if date_i < len(tags) else str(date_i),
            "selection_rule": "maximum cosine-latitude-weighted verifying ERA5 event footprint among all evaluated 2021 date/lead pairs; no forecast score used",
            "latitude_orientation": "90N to 90S",
            "longitude_orientation": "0E to 359.75E",
        }
        case_manifest[case_name] = {
            **metadata,
            "file": save_case(args.cases_dir, case_name, metadata, case_arrays[case_name]),
        }

    paired = paired_summary(bootstrap_values)
    add_controlled_router_pairs(paired, bootstrap_values)
    payload = {
        "status": "complete",
        "model": "PaE-Fuse",
        "year": 2021,
        "initialization_tags": tags,
        "date_count": n_dates,
        "lead_hours": list(TARGET_LEADS),
        "hres_contract": "corrected: HRES selected at positional old_step+1 because its coordinate includes lead 0; latitude flipped S-to-N into N-to-S",
        "threshold_contract": "all gridded tail and wind percentile thresholds were fitted from 2020 data and frozen before the independent 2021 evaluation",
        "phase_selective_gate_contract": {
            "status": "frozen before 2021 evaluation",
            "u10_formula": "sigmoid(1.5*(max(|phase-clim|,|champion-clim|)-8))*sigmoid(10*(|phase-clim|-|champion-clim|))",
            "all_variable_transfer": "same dimensionless formula using frozen L3 scales [8,8,7,1200,3000,7] and training climatology; no 2021 fitting",
        },
        "case_selection_contract": "truth-only and fixed before model inference",
        "tiling": {
            "patch": [PH, PW],
            "row_starts": rows,
            "column_starts": columns,
            "overlap_combination": "uniform average",
        },
        "resampling_unit": "initialization date",
        "bootstrap_replicates": args.bootstrap,
        "load_reports": load_reports,
        "missing_checkpoints": missing_checkpoints,
        "metrics": metrics,
        "paired_deltas": paired,
        "across_seed_summary": seed_summary({name: value["120"] for name, value in metrics.items()}),
        "across_seed_summary_scope": "120-hour lead; per-lead seed results remain in metrics",
        "phase_selective_gate_diagnostics": {
            name: {
                str(lead): finalize_gate_stats(store)
                for lead, store in by_lead.items()
            }
            for name, by_lead in selective_gate_stats.items()
        },
        "typical_cases": case_manifest,
        "coordinate_arrays": {"latitude": latitude.tolist(), "longitude": longitude.tolist()},
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
