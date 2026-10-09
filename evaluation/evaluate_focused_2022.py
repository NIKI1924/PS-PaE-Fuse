#!/usr/bin/env python3
"""Focused, sealed 2022 verification for PS-PaE-Fuse.

This evaluator intentionally covers only the predeclared 32 dates and three
lead times.  It reports the manuscript's wind endpoints, one calibrated
safeguard, deterministic baselines, paired date-bootstrap uncertainty inputs,
and a non-tuning attribution of 25 m s-1 misses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/xrx/wenqiong")

from evaluate_extreme_safeguard_2021 import apply_safeguard
from evaluate_paefuse_crossyear_2021 import (
    LEAD_INDEX,
    build_model,
    infer_full,
    load_thresholds,
    normal_crps_fast,
    phase_selective_fusion,
    sample_thresholds,
    tile_starts,
)
from evaluate_spatial_probabilistic_2021 import (
    apply_emos_wind,
    neighborhood_fraction,
    wind_probability,
)


WORK = Path("/home/xrx/wenqiong/paefuse_pr_20260904")
ROOT = Path("/vol2/xrx/paefuse_pr_20260904")
DATE_ROOT = Path("/vol2/xrx/modela_v2")
DATA_ROOT = ROOT / "focused_2022/data"
LONG_VARS = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_500",
    "temperature_850",
)
TRUTH_VARS = LONG_VARS[:4]
LEADS = (72, 120, 168)
EVENTS = ("wind_q95", "wind_q975", "wind_abs15", "wind_abs20", "wind_abs25")
PROB_EVENTS = ("wind_q95", "wind_q975")
SEEDS = (123, 456, 789)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tags_in(folder: Path) -> set[str]:
    output: set[str] = set()
    for path in folder.glob("*.npz"):
        match = re.search(r"_(\d{10})\.npz$", path.name)
        if match:
            output.add(match.group(1))
    return output


def frozen_tags() -> list[str]:
    sets = [
        tags_in(DATE_ROOT / "graphcast/2022"),
        tags_in(DATE_ROOT / "fuxi/2022"),
        tags_in(DATA_ROOT / "pangu"),
        tags_in(DATA_ROOT / "era5"),
        tags_in(DATA_ROOT / "hres"),
    ]
    common = sorted(set.intersection(*sets))
    if len(common) != 32:
        raise RuntimeError(f"sealed evaluator requires exactly 32 common dates, got {len(common)}")
    return common


def load_family(family: str, tag: str, lead_position: int) -> np.ndarray:
    if family in ("pangu", "hres"):
        path = DATA_ROOT / family / f"{family}_{tag}.npz"
        index = lead_position
    else:
        path = DATE_ROOT / family / "2022" / f"{family}_{tag}.npz"
        index = (2, 4, 6)[lead_position]
    with np.load(path) as archive:
        return np.stack(
            [np.asarray(archive[name][index], np.float32) for name in LONG_VARS]
        )


def load_truth(tag: str, lead_position: int) -> np.ndarray:
    path = DATA_ROOT / "era5" / f"era5_{tag}.npz"
    truth = np.zeros((6, 721, 1440), np.float32)
    with np.load(path) as archive:
        for index, name in enumerate(TRUTH_VARS):
            truth[index] = np.asarray(archive[name][lead_position], np.float32)
    return truth


def new_store(date_count: int, probabilistic: bool) -> dict[str, object]:
    return {
        "probabilistic": probabilistic,
        "weight": np.zeros(date_count, np.float64),
        "u10_se": np.zeros(date_count, np.float64),
        "u10_crps": np.zeros(date_count, np.float64),
        "event_counts": {event: np.zeros((date_count, 4), np.float64) for event in EVENTS},
        "fss_num": {event: np.zeros(date_count, np.float64) for event in PROB_EVENTS},
        "fss_den": {event: np.zeros(date_count, np.float64) for event in PROB_EVENTS},
        "brier": {event: np.zeros(date_count, np.float64) for event in PROB_EVENTS},
        "reliability": {
            event: {
                "weight": np.zeros((date_count, 10), np.float64),
                "probability_sum": np.zeros((date_count, 10), np.float64),
                "observed_sum": np.zeros((date_count, 10), np.float64),
            }
            for event in PROB_EVENTS
        },
    }


def threshold_map(thresholds: dict[str, object]) -> dict[str, np.ndarray | float]:
    return {
        "wind_q95": thresholds["wind_q95"],
        "wind_q975": thresholds["wind_q975"],
        "wind_abs15": 15.0,
        "wind_abs20": 20.0,
        "wind_abs25": 25.0,
    }


def update_store(
    store: dict[str, object],
    date_index: int,
    mu: np.ndarray,
    sigma: np.ndarray | None,
    truth: np.ndarray,
    thresholds: dict[str, object],
    weight: np.ndarray,
    weight_t: torch.Tensor,
    device: torch.device,
) -> None:
    error_u = mu[1] - truth[1]
    store["u10_se"][date_index] += float(np.sum(weight * error_u * error_u))
    store["weight"][date_index] += float(weight.sum())
    if sigma is not None:
        store["u10_crps"][date_index] += float(
            np.sum(weight * normal_crps_fast(mu[1], sigma[1], truth[1]))
        )

    pred_speed = np.hypot(mu[1], mu[2])
    truth_speed = np.hypot(truth[1], truth[2])
    for event, threshold in threshold_map(thresholds).items():
        forecast = pred_speed >= threshold
        observed = truth_speed >= threshold
        counts = store["event_counts"][event]
        counts[date_index, 0] += float(np.sum(weight * forecast * observed))
        counts[date_index, 1] += float(np.sum(weight * (~forecast) * observed))
        counts[date_index, 2] += float(np.sum(weight * forecast * (~observed)))
        counts[date_index, 3] += float(np.sum(weight * (~forecast) * (~observed)))
        if event not in PROB_EVENTS:
            continue
        forecast_t = torch.as_tensor(forecast.astype(np.float32), device=device)
        observed_t = torch.as_tensor(observed.astype(np.float32), device=device)
        ff = neighborhood_fraction(forecast_t, 9)
        oo = neighborhood_fraction(observed_t, 9)
        store["fss_num"][event][date_index] += float(
            (((ff - oo).square() * weight_t).sum()).item()
        )
        store["fss_den"][event][date_index] += float(
            (((ff.square() + oo.square()) * weight_t).sum()).item()
        )

    if sigma is None:
        return
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    truth_speed_t = torch.as_tensor(truth_speed, dtype=torch.float32, device=device)
    for event in PROB_EVENTS:
        threshold_t = torch.as_tensor(threshold_map(thresholds)[event], dtype=torch.float32, device=device)
        observed_t = (truth_speed_t >= threshold_t).float()
        probability = wind_probability(mu_t, sigma_t, threshold_t)
        store["brier"][event][date_index] += float(
            (((probability - observed_t).square() * weight_t).sum()).item()
        )
        bins = torch.clamp((probability * 10.0).long(), max=9).reshape(-1)
        w = weight_t.reshape(-1)
        p = probability.reshape(-1)
        o = observed_t.reshape(-1)
        rel = store["reliability"][event]
        rel["weight"][date_index] += torch.bincount(bins, weights=w, minlength=10).cpu().numpy()
        rel["probability_sum"][date_index] += torch.bincount(
            bins, weights=w * p, minlength=10
        ).cpu().numpy()
        rel["observed_sum"][date_index] += torch.bincount(
            bins, weights=w * o, minlength=10
        ).cpu().numpy()


def interval(point: float, values: np.ndarray) -> dict[str, float]:
    low, median, high = np.percentile(values, (2.5, 50.0, 97.5))
    return {
        "estimate": float(point),
        "bootstrap_median": float(median),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def event_scores(counts: np.ndarray) -> dict[str, np.ndarray]:
    h, m, f, c = (counts[..., i] for i in range(4))
    pod = h / np.maximum(h + m, 1e-12)
    false_rate = f / np.maximum(f + c, 1e-12)
    eps = 1e-6
    hh = np.clip(pod, eps, 1.0 - eps)
    ff = np.clip(false_rate, eps, 1.0 - eps)
    sedi = (
        np.log(ff) - np.log(hh) - np.log1p(-ff) + np.log1p(-hh)
    ) / (
        np.log(ff) + np.log(hh) + np.log1p(-ff) + np.log1p(-hh)
    )
    return {
        "CSI": h / np.maximum(h + m + f, 1e-12),
        "POD": pod,
        "FAR": f / np.maximum(h + f, 1e-12),
        "FBIAS": (h + f) / np.maximum(h + m, 1e-12),
        "SEDI": sedi,
    }


def summarize(
    store: dict[str, object], date_indices: np.ndarray, bootstrap: int, seed: int
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(date_indices), size=(bootstrap, len(date_indices)))
    resampled = date_indices[samples]
    weight = store["weight"]
    w0 = max(float(weight[date_indices].sum()), 1e-12)
    wb = np.maximum(weight[resampled].sum(axis=1), 1e-12)
    se = store["u10_se"]
    result: dict[str, object] = {
        "date_count": int(len(date_indices)),
        "U10_RMSE": interval(
            math.sqrt(float(se[date_indices].sum()) / w0),
            np.sqrt(se[resampled].sum(axis=1) / wb),
        ),
        "events": {},
        "FSS9": {},
    }
    if store["probabilistic"]:
        crps = store["u10_crps"]
        result["U10_CRPS"] = interval(
            float(crps[date_indices].sum()) / w0,
            crps[resampled].sum(axis=1) / wb,
        )
    for event in EVENTS:
        daily = store["event_counts"][event]
        point_scores = event_scores(daily[date_indices].sum(axis=0))
        boot_scores = event_scores(daily[resampled].sum(axis=1))
        result["events"][event] = {
            name: interval(float(point_scores[name]), boot_scores[name])
            for name in point_scores
        }
    for event in PROB_EVENTS:
        numerator = store["fss_num"][event]
        denominator = store["fss_den"][event]
        fss = 1.0 - float(numerator[date_indices].sum()) / max(
            float(denominator[date_indices].sum()), 1e-12
        )
        fss_b = 1.0 - numerator[resampled].sum(axis=1) / np.maximum(
            denominator[resampled].sum(axis=1), 1e-12
        )
        result["FSS9"][event] = interval(fss, fss_b)
    if store["probabilistic"]:
        result["Brier"] = {}
        result["reliability"] = {}
        for event in PROB_EVENTS:
            brier = store["brier"][event]
            result["Brier"][event] = interval(
                float(brier[date_indices].sum()) / w0,
                brier[resampled].sum(axis=1) / wb,
            )
            rel = store["reliability"][event]
            bin_weight = rel["weight"][date_indices].sum(axis=0)
            result["reliability"][event] = {
                "bin_edges": np.linspace(0.0, 1.0, 11).tolist(),
                "weight": bin_weight.tolist(),
                "mean_forecast_probability": np.divide(
                    rel["probability_sum"][date_indices].sum(axis=0),
                    bin_weight,
                    out=np.full(10, np.nan),
                    where=bin_weight > 0,
                ).tolist(),
                "observed_frequency": np.divide(
                    rel["observed_sum"][date_indices].sum(axis=0),
                    bin_weight,
                    out=np.full(10, np.nan),
                    where=bin_weight > 0,
                ).tolist(),
            }
    return result


def attribution_store() -> dict[str, float | list[float]]:
    return {
        "truth_weight": 0.0,
        "base_miss_weight": 0.0,
        "safeguard_miss_weight": 0.0,
        "member_ceiling_miss_weight": 0.0,
        "consensus_failure_weight": 0.0,
        "gate_active_truth_weight": 0.0,
        "gate_active_still_miss_weight": 0.0,
        "near_miss_20_to_25_weight": 0.0,
        "miss_shortfall_weighted_sum": 0.0,
        "miss_weight": 0.0,
        "selected_member_weight": [0.0, 0.0, 0.0, 0.0],
    }


def update_attribution(
    store: dict[str, float | list[float]],
    base: np.ndarray,
    safeguard: np.ndarray,
    raw: np.ndarray,
    gate: np.ndarray,
    truth: np.ndarray,
    q95: np.ndarray,
    weight: np.ndarray,
) -> None:
    members = raw.reshape(4, 6, 721, 1440)
    speeds = np.hypot(members[:, 1], members[:, 2])
    selected = speeds.argmax(axis=0)
    max_speed = speeds.max(axis=0)
    support = (speeds >= q95[None]).sum(axis=0)
    truth_speed = np.hypot(truth[1], truth[2])
    base_speed = np.hypot(base[1], base[2])
    safeguard_speed = np.hypot(safeguard[1], safeguard[2])
    truth25 = truth_speed >= 25.0
    base_miss = truth25 & (base_speed < 25.0)
    miss = truth25 & (safeguard_speed < 25.0)
    active_truth = truth25 & (gate > 0)
    store["truth_weight"] += float(np.sum(weight * truth25))
    store["base_miss_weight"] += float(np.sum(weight * base_miss))
    store["safeguard_miss_weight"] += float(np.sum(weight * miss))
    store["member_ceiling_miss_weight"] += float(np.sum(weight * truth25 * (max_speed < 25.0)))
    store["consensus_failure_weight"] += float(np.sum(weight * truth25 * (support < 2)))
    store["gate_active_truth_weight"] += float(np.sum(weight * active_truth))
    store["gate_active_still_miss_weight"] += float(np.sum(weight * active_truth * miss))
    store["near_miss_20_to_25_weight"] += float(
        np.sum(weight * miss * (safeguard_speed >= 20.0))
    )
    store["miss_shortfall_weighted_sum"] += float(
        np.sum(weight * miss * (truth_speed - safeguard_speed))
    )
    store["miss_weight"] += float(np.sum(weight * miss))
    for member in range(4):
        store["selected_member_weight"][member] += float(
            np.sum(weight * active_truth * (selected == member))
        )


def finalize_attribution(store: dict[str, float | list[float]]) -> dict[str, object]:
    truth = max(float(store["truth_weight"]), 1e-12)
    miss = max(float(store["miss_weight"]), 1e-12)
    selected = np.asarray(store["selected_member_weight"], np.float64)
    return {
        **store,
        "base_miss_fraction_of_truth25": float(store["base_miss_weight"]) / truth,
        "safeguard_miss_fraction_of_truth25": float(store["safeguard_miss_weight"]) / truth,
        "member_ceiling_fraction_of_truth25": float(store["member_ceiling_miss_weight"]) / truth,
        "consensus_failure_fraction_of_truth25": float(store["consensus_failure_weight"]) / truth,
        "gate_activation_fraction_on_truth25": float(store["gate_active_truth_weight"]) / truth,
        "near_miss_fraction_among_misses": float(store["near_miss_20_to_25_weight"]) / miss,
        "mean_speed_shortfall_on_misses_ms": float(store["miss_shortfall_weighted_sum"]) / miss,
        "selected_member_fraction_on_active_truth25": (
            selected / max(float(selected.sum()), 1e-12)
        ).tolist(),
        "member_order": ["Pangu", "GraphCast", "FuXi", "HRES"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=WORK / "focused_2022_scorecard.json")
    parser.add_argument("--cases-dir", type=Path, default=WORK / "focused_2022_cases")
    args = parser.parse_args()
    started = time.time()
    tags = frozen_tags()
    device = torch.device(f"cuda:{args.gpu}")
    thresholds_all = load_thresholds()
    latitude = np.linspace(90.0, -90.0, 721)
    weight = np.broadcast_to(
        np.maximum(np.cos(np.deg2rad(latitude)), 1e-3)[:, None], (721, 1440)
    ).copy()
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    rows, columns = tile_starts(721), tile_starts(1440)
    norm_mean = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_mean.npy").astype(np.float32)
    norm_std = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_std.npy").astype(np.float32)

    safeguard_payload = json.loads((WORK / "extreme_safeguard_2020.json").read_text(encoding="utf-8"))
    safeguard_config = safeguard_payload["selected_config"]
    calibration_payload = json.loads(
        (WORK / "focused_safeguard_scale_2020.json").read_text(encoding="utf-8")
    )
    scale = float(calibration_payload["final_scale"])
    emos_payload = json.loads((WORK / "emos_matched.json").read_text(encoding="utf-8"))
    emos_parameters = emos_payload["parameters"]

    champion_path = Path("/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt")
    champion, champion_report = build_model("base", "full", champion_path, device)
    routers: dict[int, object] = {}
    router_reports: dict[str, object] = {}
    for seed in SEEDS:
        path = ROOT / f"router_only_s{seed}/best_v3_spf.pt"
        routers[seed], router_reports[str(seed)] = build_model("paefuse", "full", path, device)

    method_probability = {
        "Pangu": False,
        "GraphCast": False,
        "FuXi": False,
        "HRES": False,
        "SimpleMean": False,
        "EMOS": True,
        "safeguard_uncalibrated_s123": True,
    }
    for seed in SEEDS:
        method_probability[f"phase_router_s{seed}"] = True
        method_probability[f"safeguard_calibrated_s{seed}"] = True
    stores = {
        method: {str(lead): new_store(len(tags), probabilistic) for lead in LEADS}
        for method, probabilistic in method_probability.items()
    }
    attribution = {
        str(seed): {str(lead): attribution_store() for lead in LEADS} for seed in SEEDS
    }
    best_case_weight = {lead: -1.0 for lead in LEADS}
    args.cases_dir.mkdir(parents=True, exist_ok=True)

    for date_index, tag in enumerate(tags):
        for lead_position, lead in enumerate(LEADS):
            members = [
                load_family("pangu", tag, lead_position),
                load_family("graphcast", tag, lead_position),
                load_family("fuxi", tag, lead_position),
                load_family("hres", tag, lead_position),
            ]
            raw = np.concatenate(members, axis=0)
            truth = load_truth(tag, lead_position)
            thresholds = sample_thresholds(thresholds_all, lead)
            norm = (raw - norm_mean[:, None, None]) / norm_std[:, None, None]
            champion_mu, champion_sigma = infer_full(
                champion, raw, norm, LEAD_INDEX[lead], rows, columns, args.batch_size, device
            )

            baseline_predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
                "Pangu": (members[0], None),
                "GraphCast": (members[1], None),
                "FuXi": (members[2], None),
                "HRES": (members[3], None),
                "SimpleMean": (np.mean(members, axis=0).astype(np.float32), None),
            }
            emos_mu, emos_sigma = apply_emos_wind(raw, emos_parameters, lead)
            baseline_predictions["EMOS"] = (emos_mu, emos_sigma)
            for method, (mu, sigma) in baseline_predictions.items():
                update_store(
                    stores[method][str(lead)], date_index, mu, sigma, truth,
                    thresholds, weight, weight_t, device,
                )

            retained: dict[str, np.ndarray] = {
                "truth": np.hypot(truth[1], truth[2]).astype(np.float32),
                "Pangu": np.hypot(members[0][1], members[0][2]).astype(np.float32),
                "SimpleMean": np.hypot(
                    baseline_predictions["SimpleMean"][0][1],
                    baseline_predictions["SimpleMean"][0][2],
                ).astype(np.float32),
            }
            for seed in SEEDS:
                router_mu, router_sigma = infer_full(
                    routers[seed], raw, norm, LEAD_INDEX[lead], rows, columns,
                    args.batch_size, device,
                )
                base_mu, base_sigma, _ = phase_selective_fusion(
                    champion_mu, champion_sigma, router_mu, router_sigma
                )
                safeguard_mu, gate = apply_safeguard(
                    base_mu, raw, thresholds, safeguard_config
                )
                calibrated_sigma = base_sigma.copy()
                calibrated_sigma[1:3] *= 1.0 + gate[None] * (scale - 1.0)
                update_store(
                    stores[f"phase_router_s{seed}"][str(lead)], date_index,
                    base_mu, base_sigma, truth, thresholds, weight, weight_t, device,
                )
                update_store(
                    stores[f"safeguard_calibrated_s{seed}"][str(lead)], date_index,
                    safeguard_mu, calibrated_sigma, truth, thresholds, weight, weight_t, device,
                )
                if seed == 123:
                    update_store(
                        stores["safeguard_uncalibrated_s123"][str(lead)], date_index,
                        safeguard_mu, base_sigma, truth, thresholds, weight, weight_t, device,
                    )
                    retained["phase_router_s123"] = np.hypot(base_mu[1], base_mu[2]).astype(np.float32)
                    retained["safeguard_s123"] = np.hypot(safeguard_mu[1], safeguard_mu[2]).astype(np.float32)
                    retained["gate"] = gate.astype(np.float32)
                update_attribution(
                    attribution[str(seed)][str(lead)], base_mu, safeguard_mu,
                    raw, gate, truth, thresholds["wind_q95"], weight,
                )

            truth_event_weight = float(
                np.sum(weight * (retained["truth"] >= thresholds["wind_q975"]))
            )
            if truth_event_weight > best_case_weight[lead]:
                best_case_weight[lead] = truth_event_weight
                np.savez_compressed(
                    args.cases_dir / f"truth_selected_q975_lead{lead}.npz",
                    **retained,
                    threshold=np.asarray(thresholds["wind_q975"], np.float32),
                    tag=np.asarray(tag),
                    lead_hour=np.asarray(lead),
                    selection_score=np.asarray(truth_event_weight),
                )
            print(f"{date_index + 1}/{len(tags)} tag={tag} lead={lead}", flush=True)

    all_dates = np.arange(len(tags))
    month = np.asarray([int(tag[4:6]) for tag in tags])
    seasons = {
        "DJF": np.where(np.isin(month, (12, 1, 2)))[0],
        "MAM": np.where(np.isin(month, (3, 4, 5)))[0],
        "JJA": np.where(np.isin(month, (6, 7, 8)))[0],
        "SON": np.where(np.isin(month, (9, 10, 11)))[0],
    }
    metrics: dict[str, object] = {}
    for method, lead_stores in stores.items():
        metrics[method] = {"annual": {}, "seasonal": {}}
        for lead in LEADS:
            metrics[method]["annual"][str(lead)] = summarize(
                lead_stores[str(lead)], all_dates, args.bootstrap, 20260913 + lead
            )
        for season, indices in seasons.items():
            metrics[method]["seasonal"][season] = {
                str(lead): summarize(
                    lead_stores[str(lead)], indices, min(args.bootstrap, 1000),
                    20260913 + lead + int(indices[0]),
                )
                for lead in LEADS
            }

    payload = {
        "status": "complete_and_sealed",
        "year": 2022,
        "date_tags": tags,
        "lead_hours": list(LEADS),
        "methods": list(stores),
        "safeguard_config": safeguard_config,
        "conditional_scale": scale,
        "metrics": metrics,
        "attribution_25ms": {
            seed: {lead: finalize_attribution(value) for lead, value in by_lead.items()}
            for seed, by_lead in attribution.items()
        },
        "test_contract": "32 fixed dates; no 2022 score entered model, gate, calibration, threshold, or event-definition selection",
        "bootstrap_replicates": args.bootstrap,
        "resampling_unit": "initialization date",
        "checkpoint_load_reports": {"champion": champion_report, "routers": router_reports},
        "frozen_input_sha256": {
            "protocol": sha256(WORK / "FOCUSED_2022_FROZEN_PROTOCOL.md"),
            "calibration": sha256(WORK / "focused_safeguard_scale_2020.json"),
            "event_definitions": sha256(WORK / "focused_event_definitions_2022.json"),
            "evaluator": sha256(Path(__file__)),
            "safeguard_selection": sha256(WORK / "extreme_safeguard_2020.json"),
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
