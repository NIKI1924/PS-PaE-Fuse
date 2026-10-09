#!/usr/bin/env python3
"""Compute q97.5 FSS9 for the four raw members and their simple mean.

The calculation uses exactly the 22-date, 72/120/168-h 2021 evaluation
contract used by the PS-PaE-Fuse manuscript: the cached Pangu-Weather,
GraphCast and FuXi fields, aligned HRES fields, frozen 2020 gridwise q97.5
thresholds, periodic longitude, replicated latitude, and cosine-latitude
weighting.  The script performs no model inference and writes one compact
JSON evidence file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_paefuse_crossyear_2021 import (
    LONG_VARS,
    TARGET_LEADS,
    VARS,
    load_thresholds,
    sample_thresholds,
)


METHODS = ("HRES", "Pangu-Weather", "GraphCast", "FuXi", "Simple mean")
WINDOW = 9


def neighbourhood_fraction(mask: torch.Tensor, window: int = WINDOW) -> torch.Tensor:
    pad = window // 2
    field = F.pad(mask[None, None], (pad, pad, 0, 0), mode="circular")
    field = F.pad(field, (0, 0, pad, pad), mode="replicate")
    return F.avg_pool2d(field, kernel_size=window, stride=1)[0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xrx/wenqiong/paefuse_pr_20260904/raw_member_fss9_2021.json"),
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cache_root = Path("/vol2/xrx/baseline_cache_2021")
    hres_root = Path("/vol2/xrx/wb2_2021_unified/hres")
    caches = [np.load(cache_root / f"cache_{name}.npy", mmap_mode="r") for name in VARS]
    hres = [np.load(hres_root / f"{name}.npy", mmap_mode="r") for name in LONG_VARS]
    meta = np.load(cache_root / "sample_meta.npy")
    thresholds_all = load_thresholds()
    selected = [index for index, row in enumerate(meta) if int(row[3]) in TARGET_LEADS]

    latitude = np.linspace(90.0, -90.0, 721)
    area_weight = torch.as_tensor(
        np.broadcast_to(np.maximum(np.cos(np.deg2rad(latitude)), 1e-3)[:, None], (721, 1440)).copy(),
        dtype=torch.float32,
        device=device,
    )
    totals = {
        method: {str(lead): {"numerator": 0.0, "denominator": 0.0, "cases": 0} for lead in TARGET_LEADS}
        for method in METHODS
    }

    for count, sample_index in enumerate(selected, start=1):
        date_index, old_hres_step, _, lead = map(int, meta[sample_index])
        raw = np.empty((4, 6, 721, 1440), np.float32)
        truth = np.empty((6, 721, 1440), np.float32)
        for variable_index in range(6):
            raw[0, variable_index] = np.asarray(caches[variable_index][sample_index, 0], np.float32)
            raw[1, variable_index] = np.asarray(caches[variable_index][sample_index, 1], np.float32)
            raw[2, variable_index] = np.asarray(caches[variable_index][sample_index, 2], np.float32)
            raw[3, variable_index] = np.asarray(
                hres[variable_index][date_index, old_hres_step + 1, ::-1, :], np.float32
            )
            truth[variable_index] = np.asarray(caches[variable_index][sample_index, 4], np.float32)

        threshold = torch.as_tensor(
            sample_thresholds(thresholds_all, lead)["wind_q975"],
            dtype=torch.float32,
            device=device,
        )
        truth_t = torch.as_tensor(truth, dtype=torch.float32, device=device)
        observed = (torch.hypot(truth_t[1], truth_t[2]) >= threshold).float()
        observed_fraction = neighbourhood_fraction(observed)

        fields = {
            "HRES": raw[3],
            "Pangu-Weather": raw[0],
            "GraphCast": raw[1],
            "FuXi": raw[2],
            "Simple mean": raw.mean(axis=0),
        }
        for method, field in fields.items():
            field_t = torch.as_tensor(field, dtype=torch.float32, device=device)
            forecast = (torch.hypot(field_t[1], field_t[2]) >= threshold).float()
            forecast_fraction = neighbourhood_fraction(forecast)
            numerator = ((forecast_fraction - observed_fraction).square() * area_weight).sum()
            denominator = ((forecast_fraction.square() + observed_fraction.square()) * area_weight).sum()
            bucket = totals[method][str(lead)]
            bucket["numerator"] += float(numerator.item())
            bucket["denominator"] += float(denominator.item())
            bucket["cases"] += 1
        print(f"{count}/{len(selected)} date={date_index} lead={lead}", flush=True)

    results = {}
    for method, by_lead in totals.items():
        lead_values = {}
        for lead, bucket in by_lead.items():
            lead_values[lead] = 1.0 - bucket["numerator"] / max(bucket["denominator"], 1e-12)
        results[method] = {
            "by_lead": lead_values,
            "mean_of_lead_estimates": float(np.mean(list(lead_values.values()))),
        }

    payload = {
        "status": "complete",
        "year": 2021,
        "date_count": int(meta[:, 0].max()) + 1,
        "lead_hours": list(TARGET_LEADS),
        "event": "wind_q975",
        "window_grid_cells": WINDOW,
        "boundary_contract": "periodic longitude, replicated latitude; cosine-latitude weighting",
        "aggregation": "FSS pooled over dates within lead, then arithmetic mean over 72/120/168 h",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
