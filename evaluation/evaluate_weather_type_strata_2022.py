#!/usr/bin/env python3
"""Frozen weather-type stratification of the sealed 2022 PS-PaE-Fuse sample.

The script reuses the 32 fixed initialization dates and all frozen model,
threshold, safeguard and calibration choices.  It implements the definitions in
``focused_event_definitions_2022.json`` without tuning on 2022 scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.special import erfc

sys.path.insert(0, "/home/xrx/wenqiong")

from evaluate_extreme_safeguard_2021 import apply_safeguard
from evaluate_focused_2022 import (
    DATA_ROOT,
    DATE_ROOT,
    LEADS,
    ROOT,
    SEEDS,
    WORK,
    frozen_tags,
    load_family,
    load_truth,
)
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
)


IBTRACS = WORK / "ibtracs_2022_v04r01.csv"
EVENT_DEFINITIONS = WORK / "focused_event_definitions_2022.json"
PROTOCOL = WORK / "WEATHER_TYPE_STRATIFICATION_PROTOCOL_2022.md"
EARTH_RADIUS_KM = 6371.0088
TC_RADIUS_KM = 500.0
WINDOW = 9
STRATA = ("tropical_cyclone", "cold_surge", "non_tropical_high_wind")
PROBABILISTIC_PREFIXES = ("EMOS", "phase_router_", "safeguard_calibrated_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tag(tag: str) -> datetime:
    return datetime.strptime(tag, "%Y%m%d%H").replace(tzinfo=timezone.utc)


def read_ibtracs(path: Path) -> dict[str, list[dict[str, object]]]:
    """Read the NOAA subset and index unique storm records by exact UTC time."""
    records: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("time") == "UTC":  # ERDDAP units row
                continue
            sid = (row.get("sid") or "").strip()
            stamp = (row.get("time") or "").strip()
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"]) % 360.0
            except (KeyError, TypeError, ValueError):
                continue
            key = (sid, stamp)
            if not sid or not stamp or key in seen:
                continue
            seen.add(key)
            records[stamp].append(
                {
                    "sid": sid,
                    "name": (row.get("name") or "NOT_NAMED").strip(),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return records


def great_circle_indices(
    centre_latitude: float,
    centre_longitude: float,
    latitude_radians: np.ndarray,
    longitude_radians: np.ndarray,
) -> np.ndarray:
    """Return flat grid indices no farther than 500 km from one storm centre."""
    lat0 = math.radians(centre_latitude)
    lon0 = math.radians(centre_longitude)
    cosine = (
        np.sin(latitude_radians)[:, None] * math.sin(lat0)
        + np.cos(latitude_radians)[:, None]
        * math.cos(lat0)
        * np.cos(longitude_radians[None, :] - lon0)
    )
    distance = EARTH_RADIUS_KM * np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.flatnonzero(distance.reshape(-1) <= TC_RADIUS_KM)


def new_accumulator(probabilistic: bool) -> dict[str, float | bool]:
    return {
        "probabilistic": probabilistic,
        "domain_weight": 0.0,
        "continuous_weight": 0.0,
        "observed_event_weight": 0.0,
        "u10_se": 0.0,
        "u10_crps": 0.0,
        "tp": 0.0,
        "fn": 0.0,
        "fp": 0.0,
        "tn": 0.0,
        "fss_num": 0.0,
        "fss_den": 0.0,
        "brier": 0.0,
    }


def probabilistic_method(name: str) -> bool:
    return name == "EMOS" or any(name.startswith(prefix) for prefix in PROBABILISTIC_PREFIXES[1:])


def wind_probability_numpy(
    mu: np.ndarray, sigma: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    u, v = mu[1], mu[2]
    su = np.maximum(sigma[1], 0.02)
    sv = np.maximum(sigma[2], 0.02)
    speed = np.hypot(u, v)
    speed_sd = np.sqrt((u * su) ** 2 + (v * sv) ** 2) / np.maximum(speed, 0.5)
    speed_sd = np.maximum(speed_sd, 0.05)
    z = (threshold - speed) / (math.sqrt(2.0) * speed_sd)
    return np.clip(0.5 * erfc(z), 0.0, 1.0).astype(np.float32)


def metric_fields(
    mu: np.ndarray,
    sigma: np.ndarray | None,
    truth: np.ndarray,
    threshold: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray | None]:
    forecast_speed = np.hypot(mu[1], mu[2])
    truth_speed = np.hypot(truth[1], truth[2])
    forecast = forecast_speed >= threshold
    observed = truth_speed >= threshold
    forecast_t = torch.as_tensor(forecast.astype(np.float32), device=device)
    observed_t = torch.as_tensor(observed.astype(np.float32), device=device)
    forecast_fraction = neighborhood_fraction(forecast_t, WINDOW)
    observed_fraction = neighborhood_fraction(observed_t, WINDOW)
    fss_num = (forecast_fraction - observed_fraction).square().cpu().numpy()
    fss_den = (forecast_fraction.square() + observed_fraction.square()).cpu().numpy()
    probability: np.ndarray | None = None
    crps: np.ndarray | None = None
    if sigma is not None:
        probability = wind_probability_numpy(mu, sigma, threshold)
        crps = normal_crps_fast(mu[1], sigma[1], truth[1]).astype(np.float32)
    return {
        "u10_se": ((mu[1] - truth[1]) ** 2).astype(np.float32),
        "crps": crps,
        "forecast": forecast,
        "observed": observed,
        "fss_num": fss_num,
        "fss_den": fss_den,
        "probability": probability,
    }


def add_fields(
    accumulator: dict[str, float | bool],
    fields: dict[str, np.ndarray | None],
    domain_indices: np.ndarray,
    area_weight_flat: np.ndarray,
    continuous_event_only: bool,
) -> None:
    if domain_indices.size == 0:
        return
    weights = area_weight_flat[domain_indices]
    observed = np.asarray(fields["observed"]).reshape(-1)[domain_indices]
    forecast = np.asarray(fields["forecast"]).reshape(-1)[domain_indices]
    if continuous_event_only:
        continuous_local = observed
    else:
        continuous_local = np.ones(observed.shape, dtype=bool)
    continuous_indices = domain_indices[continuous_local]
    continuous_weights = area_weight_flat[continuous_indices]

    accumulator["domain_weight"] += float(weights.sum())
    accumulator["continuous_weight"] += float(continuous_weights.sum())
    accumulator["observed_event_weight"] += float(weights[observed].sum())
    accumulator["tp"] += float(weights[forecast & observed].sum())
    accumulator["fn"] += float(weights[(~forecast) & observed].sum())
    accumulator["fp"] += float(weights[forecast & (~observed)].sum())
    accumulator["tn"] += float(weights[(~forecast) & (~observed)].sum())
    accumulator["fss_num"] += float(
        (weights * np.asarray(fields["fss_num"]).reshape(-1)[domain_indices]).sum()
    )
    accumulator["fss_den"] += float(
        (weights * np.asarray(fields["fss_den"]).reshape(-1)[domain_indices]).sum()
    )
    if continuous_indices.size:
        accumulator["u10_se"] += float(
            (
                continuous_weights
                * np.asarray(fields["u10_se"]).reshape(-1)[continuous_indices]
            ).sum()
        )
    crps = fields["crps"]
    probability = fields["probability"]
    if crps is not None and continuous_indices.size:
        accumulator["u10_crps"] += float(
            (continuous_weights * np.asarray(crps).reshape(-1)[continuous_indices]).sum()
        )
    if probability is not None:
        probability_values = np.asarray(probability).reshape(-1)[domain_indices]
        accumulator["brier"] += float(
            (weights * (probability_values - observed.astype(np.float32)) ** 2).sum()
        )


def sum_accumulators(values: Iterable[dict[str, float | bool]]) -> dict[str, float | bool]:
    output = new_accumulator(False)
    probabilistic = False
    for value in values:
        probabilistic = probabilistic or bool(value["probabilistic"])
        for key in output:
            if key == "probabilistic":
                continue
            output[key] = float(output[key]) + float(value[key])
    output["probabilistic"] = probabilistic
    return output


def score(accumulator: dict[str, float | bool], metric: str) -> float:
    eps = 1e-12
    if metric == "U10_RMSE":
        return math.sqrt(float(accumulator["u10_se"]) / max(float(accumulator["continuous_weight"]), eps))
    if metric == "U10_CRPS":
        return float(accumulator["u10_crps"]) / max(float(accumulator["continuous_weight"]), eps)
    if metric == "CSI":
        return float(accumulator["tp"]) / max(
            float(accumulator["tp"]) + float(accumulator["fn"]) + float(accumulator["fp"]), eps
        )
    if metric == "POD":
        return float(accumulator["tp"]) / max(float(accumulator["tp"]) + float(accumulator["fn"]), eps)
    if metric == "FAR":
        return float(accumulator["fp"]) / max(float(accumulator["tp"]) + float(accumulator["fp"]), eps)
    if metric == "FSS9":
        return 1.0 - float(accumulator["fss_num"]) / max(float(accumulator["fss_den"]), eps)
    if metric == "Brier":
        return float(accumulator["brier"]) / max(float(accumulator["domain_weight"]), eps)
    raise KeyError(metric)


def quantile_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], np.float64)
    if array.size == 0:
        return {"bootstrap_median": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {
        "bootstrap_median": float(np.quantile(array, 0.5)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def summarize_units(
    units: dict[str, dict[str, float | bool]], bootstrap: int, random_seed: int
) -> dict[str, object]:
    identifiers = sorted(units)
    if not identifiers:
        return {
            "unit_count": 0,
            "event_unit_count": 0,
            "inferential_status": "descriptive_only",
            "metrics": {},
        }
    total = sum_accumulators(units[identifier] for identifier in identifiers)
    event_units = sum(float(units[identifier]["observed_event_weight"]) > 0.0 for identifier in identifiers)
    metrics = ["U10_RMSE", "CSI", "POD", "FAR", "FSS9"]
    if bool(total["probabilistic"]):
        metrics.extend(("U10_CRPS", "Brier"))
    rng = np.random.default_rng(random_seed)
    samples = rng.integers(0, len(identifiers), size=(bootstrap, len(identifiers)))
    output_metrics: dict[str, object] = {}
    for metric in metrics:
        replicates: list[float] = []
        for row in samples:
            replicate = sum_accumulators(units[identifiers[index]] for index in row)
            replicates.append(score(replicate, metric))
        output_metrics[metric] = {"estimate": score(total, metric), **quantile_interval(replicates)}
    return {
        "unit_count": len(identifiers),
        "event_unit_count": int(event_units),
        "inferential_status": "inferential" if event_units >= 10 else "descriptive_only",
        "domain_weight": float(total["domain_weight"]),
        "observed_event_weight": float(total["observed_event_weight"]),
        "unit_ids": identifiers,
        "metrics": output_metrics,
    }


def paired_comparison(
    baseline: dict[str, dict[str, float | bool]],
    candidate: dict[str, dict[str, float | bool]],
    metrics: tuple[str, ...],
    bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    identifiers = sorted(set(baseline) & set(candidate))
    if not identifiers:
        return {"unit_count": 0, "metrics": {}}
    base_total = sum_accumulators(baseline[identifier] for identifier in identifiers)
    candidate_total = sum_accumulators(candidate[identifier] for identifier in identifiers)
    rng = np.random.default_rng(random_seed)
    samples = rng.integers(0, len(identifiers), size=(bootstrap, len(identifiers)))
    output: dict[str, object] = {}
    for metric in metrics:
        base_point = score(base_total, metric)
        candidate_point = score(candidate_total, metric)
        differences: list[float] = []
        for row in samples:
            base_replicate = sum_accumulators(baseline[identifiers[index]] for index in row)
            candidate_replicate = sum_accumulators(candidate[identifiers[index]] for index in row)
            differences.append(score(candidate_replicate, metric) - score(base_replicate, metric))
        if metric in ("U10_RMSE", "U10_CRPS", "Brier"):
            improvement = 100.0 * (base_point - candidate_point) / max(abs(base_point), 1e-12)
        else:
            improvement = 100.0 * (candidate_point - base_point) / max(abs(base_point), 1e-12)
        output[metric] = {
            "baseline": base_point,
            "candidate": candidate_point,
            "candidate_minus_baseline": candidate_point - base_point,
            "directional_improvement_percent": improvement,
            **quantile_interval(differences),
        }
    return {"unit_count": len(identifiers), "metrics": output}


def save_case(
    output_dir: Path,
    stratum: str,
    score_value: float,
    tag: str,
    lead: int,
    domain_indices: np.ndarray,
    retained: dict[str, np.ndarray],
    threshold: np.ndarray,
    metadata: dict[str, object],
) -> str:
    domain = np.zeros((721 * 1440,), dtype=np.uint8)
    domain[domain_indices] = 1
    path = output_dir / f"{stratum}_truth_selected.npz"
    serializable = {
        **retained,
        "domain_mask": domain.reshape(721, 1440),
        "threshold": np.asarray(threshold, np.float32),
        "tag": np.asarray(tag),
        "lead_hour": np.asarray(lead),
        "selection_score": np.asarray(score_value),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    np.savez_compressed(path, **serializable)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=WORK / "weather_type_strata_2022.json")
    parser.add_argument("--cases-dir", type=Path, default=WORK / "weather_type_cases_2022")
    args = parser.parse_args()
    started = time.time()

    definitions = json.loads(EVENT_DEFINITIONS.read_text(encoding="utf-8"))
    if definitions.get("status") != "frozen_before_2022_scores":
        raise RuntimeError("weather-type definitions are not marked frozen")
    tags = frozen_tags()
    ibtracs_by_time = read_ibtracs(IBTRACS)
    expected_valid_times = [
        (parse_tag(tag) + timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        for tag in tags
        for lead in LEADS
    ]
    matched_track_times = [stamp for stamp in expected_valid_times if stamp in ibtracs_by_time]
    matched_track_records = sum(len(ibtracs_by_time[stamp]) for stamp in matched_track_times)
    if matched_track_records == 0:
        raise RuntimeError(
            "none of the 96 frozen valid times matched IBTrACS; check timestamp format and subset"
        )
    device = torch.device(f"cuda:{args.gpu}")
    thresholds_all = load_thresholds()
    t2m_p05 = np.asarray(thresholds_all["tails"]["T2M"]["p05"], np.float32)

    latitude = np.linspace(90.0, -90.0, 721)
    longitude = np.linspace(0.0, 359.75, 1440)
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    area_weight = np.broadcast_to(
        np.maximum(np.cos(latitude_radians), 1e-3)[:, None], (721, 1440)
    ).copy()
    area_weight_flat = area_weight.reshape(-1)
    rows, columns = tile_starts(721), tile_starts(1440)
    norm_mean = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_mean.npy").astype(np.float32)
    norm_std = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_std.npy").astype(np.float32)

    safeguard_payload = json.loads((WORK / "extreme_safeguard_2020.json").read_text(encoding="utf-8"))
    safeguard_config = safeguard_payload["selected_config"]
    calibration_payload = json.loads((WORK / "focused_safeguard_scale_2020.json").read_text(encoding="utf-8"))
    scale = float(calibration_payload["final_scale"])
    emos_payload = json.loads((WORK / "emos_matched.json").read_text(encoding="utf-8"))
    emos_parameters = emos_payload["parameters"]

    champion_path = Path("/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt")
    champion, champion_report = build_model("base", "full", champion_path, device)
    routers: dict[int, object] = {}
    router_reports: dict[str, object] = {}
    for seed in SEEDS:
        router_path = ROOT / f"router_only_s{seed}/best_v3_spf.pt"
        routers[seed], router_reports[str(seed)] = build_model("paefuse", "full", router_path, device)

    method_names = ["Pangu", "GraphCast", "FuXi", "HRES", "SimpleMean", "EMOS"]
    for seed in SEEDS:
        method_names.extend((f"phase_router_s{seed}", f"safeguard_calibrated_s{seed}"))
    units: dict[str, dict[str, dict[str, dict[str, float | bool]]]] = {
        stratum: {method: {} for method in method_names} for stratum in STRATA
    }
    stratum_occurrences = {stratum: 0 for stratum in STRATA}
    independent_tc_metadata: dict[str, dict[str, object]] = {}
    best_case = {stratum: {"score": -1.0, "path": None} for stratum in STRATA}
    args.cases_dir.mkdir(parents=True, exist_ok=True)

    def update(
        stratum: str,
        method: str,
        unit_id: str,
        fields: dict[str, np.ndarray | None],
        indices: np.ndarray,
        continuous_event_only: bool,
    ) -> None:
        if unit_id not in units[stratum][method]:
            units[stratum][method][unit_id] = new_accumulator(probabilistic_method(method))
        add_fields(
            units[stratum][method][unit_id], fields, indices,
            area_weight_flat, continuous_event_only,
        )

    for date_index, tag in enumerate(tags):
        truth_by_lead = {
            lead: load_truth(tag, lead_position) for lead_position, lead in enumerate(LEADS)
        }
        cold_masks = {
            120: (truth_by_lead[120][0] - truth_by_lead[72][0] <= -6.0)
            & (truth_by_lead[120][0] <= t2m_p05),
            168: (truth_by_lead[168][0] - truth_by_lead[120][0] <= -6.0)
            & (truth_by_lead[168][0] <= t2m_p05),
        }

        for lead_position, lead in enumerate(LEADS):
            valid_time = parse_tag(tag) + timedelta(hours=lead)
            # NOAA ERDDAP emits ISO-8601 timestamps with an explicit seconds
            # field (for example ``2022-01-08T00:00:00Z``).  Keep that exact
            # representation so the frozen forecast valid times actually
            # cross-match the IBTrACS records.
            stamp = valid_time.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            storm_entries: list[tuple[str, str, np.ndarray, float, float]] = []
            tc_union = np.zeros((721 * 1440,), dtype=bool)
            for record in ibtracs_by_time.get(stamp, []):
                storm_indices = great_circle_indices(
                    float(record["latitude"]), float(record["longitude"]),
                    latitude_radians, longitude_radians,
                )
                if storm_indices.size == 0:
                    continue
                tc_union[storm_indices] = True
                sid = str(record["sid"])
                name = str(record["name"])
                storm_entries.append(
                    (sid, name, storm_indices, float(record["latitude"]), float(record["longitude"]))
                )
                independent_tc_metadata.setdefault(sid, {"name": name, "sid": sid})

            non_tc_indices = np.flatnonzero(~tc_union)
            cold_indices = (
                np.flatnonzero(cold_masks[lead].reshape(-1)) if lead in cold_masks else np.empty(0, np.int64)
            )
            stratum_occurrences["tropical_cyclone"] += len(storm_entries)
            stratum_occurrences["cold_surge"] += int(cold_indices.size > 0)
            stratum_occurrences["non_tropical_high_wind"] += 1

            members = [
                load_family("pangu", tag, lead_position),
                load_family("graphcast", tag, lead_position),
                load_family("fuxi", tag, lead_position),
                load_family("hres", tag, lead_position),
            ]
            raw = np.concatenate(members, axis=0)
            truth = truth_by_lead[lead]
            thresholds = sample_thresholds(thresholds_all, lead)
            q975 = np.asarray(thresholds["wind_q975"], np.float32)
            q975_and_20 = np.maximum(q975, 20.0).astype(np.float32)
            norm = (raw - norm_mean[:, None, None]) / norm_std[:, None, None]
            champion_mu, champion_sigma = infer_full(
                champion, raw, norm, LEAD_INDEX[lead], rows, columns,
                args.batch_size, device,
            )

            simple_mean = np.mean(members, axis=0).astype(np.float32)
            emos_mu, emos_sigma = apply_emos_wind(raw, emos_parameters, lead)
            predictions: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
                "Pangu": (members[0], None),
                "GraphCast": (members[1], None),
                "FuXi": (members[2], None),
                "HRES": (members[3], None),
                "SimpleMean": (simple_mean, None),
                "EMOS": (emos_mu, emos_sigma),
            }
            retained: dict[str, np.ndarray] = {
                "truth": np.hypot(truth[1], truth[2]).astype(np.float32),
                "Pangu": np.hypot(members[0][1], members[0][2]).astype(np.float32),
                "FuXi": np.hypot(members[2][1], members[2][2]).astype(np.float32),
                "SimpleMean": np.hypot(simple_mean[1], simple_mean[2]).astype(np.float32),
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
                predictions[f"phase_router_s{seed}"] = (base_mu, base_sigma)
                predictions[f"safeguard_calibrated_s{seed}"] = (
                    safeguard_mu, calibrated_sigma
                )
                if seed == 123:
                    retained["phase_router_s123"] = np.hypot(base_mu[1], base_mu[2]).astype(np.float32)
                    retained["safeguard_s123"] = np.hypot(safeguard_mu[1], safeguard_mu[2]).astype(np.float32)
                    retained["gate"] = gate.astype(np.float32)

            truth_speed = retained["truth"]
            truth_q975 = truth_speed >= q975
            truth_non_tc = truth_speed >= q975_and_20
            for method, (mu, sigma) in predictions.items():
                standard_fields = metric_fields(mu, sigma, truth, q975, device)
                for sid, _name, indices, _lat, _lon in storm_entries:
                    update("tropical_cyclone", method, sid, standard_fields, indices, False)
                if cold_indices.size:
                    update("cold_surge", method, tag, standard_fields, cold_indices, False)
                del standard_fields

                non_tc_fields = metric_fields(mu, sigma, truth, q975_and_20, device)
                update(
                    "non_tropical_high_wind", method, tag,
                    non_tc_fields, non_tc_indices, True,
                )
                del non_tc_fields

            for sid, name, indices, centre_lat, centre_lon in storm_entries:
                selection_score = float(area_weight_flat[indices][truth_q975.reshape(-1)[indices]].sum())
                if selection_score > float(best_case["tropical_cyclone"]["score"]):
                    path = save_case(
                        args.cases_dir, "tropical_cyclone", selection_score,
                        tag, lead, indices, retained, q975,
                        {
                            "sid": sid,
                            "name": name,
                            "centre_latitude": centre_lat,
                            "centre_longitude": centre_lon,
                            "valid_time": stamp,
                            "selection": "largest truth-only q97.5 event area within 500 km",
                        },
                    )
                    best_case["tropical_cyclone"] = {"score": selection_score, "path": path}

            if cold_indices.size:
                selection_score = float(
                    area_weight_flat[cold_indices][truth_q975.reshape(-1)[cold_indices]].sum()
                )
                if selection_score > float(best_case["cold_surge"]["score"]):
                    path = save_case(
                        args.cases_dir, "cold_surge", selection_score,
                        tag, lead, cold_indices, retained, q975,
                        {
                            "valid_time": stamp,
                            "selection": "largest truth-only q97.5 wind area inside frozen cold-surge mask",
                            "temperature_drop_hours": 48,
                        },
                    )
                    best_case["cold_surge"] = {"score": selection_score, "path": path}

            non_tc_selection = non_tc_indices[truth_non_tc.reshape(-1)[non_tc_indices]]
            selection_score = float(area_weight_flat[non_tc_selection].sum())
            if selection_score > float(best_case["non_tropical_high_wind"]["score"]):
                path = save_case(
                    args.cases_dir, "non_tropical_high_wind", selection_score,
                    tag, lead, non_tc_indices, retained, q975_and_20,
                    {
                        "valid_time": stamp,
                        "selection": "largest truth-only non-tropical q97.5 and 20 m s-1 event area",
                    },
                )
                best_case["non_tropical_high_wind"] = {"score": selection_score, "path": path}

            print(
                f"{date_index + 1}/{len(tags)} tag={tag} lead={lead} "
                f"storms={len(storm_entries)} cold_cells={cold_indices.size}",
                flush=True,
            )

    summaries: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    seed_stability: dict[str, object] = {}
    for stratum_index, stratum in enumerate(STRATA):
        summaries[stratum] = {
            method: summarize_units(
                units[stratum][method], args.bootstrap,
                20260914 + stratum_index * 100 + method_index,
            )
            for method_index, method in enumerate(method_names)
        }
        comparisons[stratum] = {
            "safeguard_s123_vs_Pangu": paired_comparison(
                units[stratum]["Pangu"], units[stratum]["safeguard_calibrated_s123"],
                ("U10_RMSE", "CSI", "POD", "FAR", "FSS9"),
                args.bootstrap, 20261914 + stratum_index,
            ),
            "safeguard_s123_vs_phase_router_s123": paired_comparison(
                units[stratum]["phase_router_s123"],
                units[stratum]["safeguard_calibrated_s123"],
                ("U10_RMSE", "U10_CRPS", "CSI", "POD", "FAR", "FSS9", "Brier"),
                args.bootstrap, 20262914 + stratum_index,
            ),
        }
        stability: dict[str, object] = {}
        for output_prefix in ("phase_router_s", "safeguard_calibrated_s"):
            stability[output_prefix.rstrip("_s")] = {}
            for metric in ("U10_RMSE", "CSI", "FSS9", "U10_CRPS", "Brier"):
                values = []
                for seed in SEEDS:
                    result = summaries[stratum][f"{output_prefix}{seed}"]
                    if metric in result["metrics"]:
                        values.append(float(result["metrics"][metric]["estimate"]))
                if values:
                    stability[output_prefix.rstrip("_s")][metric] = {
                        "minimum": min(values),
                        "maximum": max(values),
                        "range": max(values) - min(values),
                    }
        seed_stability[stratum] = stability

    event_counts = {
        stratum: {
            "occurrence_count": int(stratum_occurrences[stratum]),
            "independent_unit_count": int(summaries[stratum]["Pangu"]["unit_count"]),
            "independent_event_unit_count": int(summaries[stratum]["Pangu"]["event_unit_count"]),
            "inferential_status": summaries[stratum]["Pangu"]["inferential_status"],
        }
        for stratum in STRATA
    }
    payload = {
        "status": "complete_frozen_weather_type_stratification",
        "year": 2022,
        "date_tags": tags,
        "lead_hours": list(LEADS),
        "methods": method_names,
        "event_counts": event_counts,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "seed_stability": seed_stability,
        "unit_accumulators": units,
        "typical_cases": best_case,
        "tropical_cyclone_systems": sorted(independent_tc_metadata.values(), key=lambda value: str(value["sid"])),
        "ibtracs_crossmatch": {
            "frozen_valid_time_count": len(expected_valid_times),
            "valid_times_with_at_least_one_track": len(matched_track_times),
            "matched_track_record_count": int(matched_track_records),
        },
        "definitions": definitions,
        "statistical_contract": {
            "bootstrap_replicates": args.bootstrap,
            "minimum_event_units_for_inference": 10,
            "tropical_cyclone_resampling_unit": "unique IBTrACS SID",
            "cold_surge_resampling_unit": "initialization date",
            "non_tropical_high_wind_resampling_unit": "initialization date",
            "case_selection": "truth-only event area; no relative model score",
            "scope": "32 fixed dates, not an exhaustive annual event catalogue",
        },
        "checkpoint_load_reports": {"champion": champion_report, "routers": router_reports},
        "frozen_input_sha256": {
            "protocol": sha256(PROTOCOL),
            "event_definitions": sha256(EVENT_DEFINITIONS),
            "ibtracs_subset": sha256(IBTRACS),
            "evaluator": sha256(Path(__file__)),
            "safeguard_selection": sha256(WORK / "extreme_safeguard_2020.json"),
            "conditional_scale": sha256(WORK / "focused_safeguard_scale_2020.json"),
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
