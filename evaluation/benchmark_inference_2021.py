#!/usr/bin/env python3
"""Reproducible full-field inference benchmark for the deployable phase router."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_paefuse_crossyear_2021 import (
    LEAD_INDEX,
    LONG_VARS,
    VARS,
    build_model,
    infer_full,
    phase_selective_fusion,
    tile_starts,
)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lead", type=int, default=120, choices=(72, 120, 168))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xrx/wenqiong/paefuse_pr_20260904/inference_benchmark.json"),
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    cache_root = Path("/vol2/xrx/baseline_cache_2021")
    hres_root = Path("/vol2/xrx/wb2_2021_unified/hres")
    caches = [np.load(cache_root / f"cache_{name}.npy", mmap_mode="r") for name in VARS]
    hres = [np.load(hres_root / f"{name}.npy", mmap_mode="r") for name in LONG_VARS]
    meta = np.load(cache_root / "sample_meta.npy")
    sample_index = next(i for i, row in enumerate(meta) if int(row[3]) == args.lead)
    date_index, old_hres_step, _, lead = map(int, meta[sample_index])
    raw = np.empty((24, 721, 1440), np.float32)
    for variable_index in range(6):
        raw[variable_index] = np.asarray(caches[variable_index][sample_index, 0], np.float32)
        raw[6 + variable_index] = np.asarray(caches[variable_index][sample_index, 1], np.float32)
        raw[12 + variable_index] = np.asarray(caches[variable_index][sample_index, 2], np.float32)
        raw[18 + variable_index] = np.asarray(
            hres[variable_index][date_index, old_hres_step + 1, ::-1, :], np.float32
        )
    norm_mean = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_mean.npy").astype(np.float32)
    norm_std = np.load("/home/xrx/wenqiong/patch_cache_7lt/norm_std.npy").astype(np.float32)
    norm = (raw - norm_mean[:, None, None]) / norm_std[:, None, None]
    rows, columns = tile_starts(721), tile_starts(1440)

    champion, champion_report = build_model(
        "base", "full",
        Path("/home/xrx/wenqiong/ov2_dem_c32_s123_u10fix/best_v3_spf.pt"),
        device,
    )
    router, router_report = build_model(
        "paefuse", "full",
        Path("/vol2/xrx/paefuse_pr_20260904/router_only_s123/best_v3_spf.pt"),
        device,
    )

    def run_once() -> None:
        champion_mu, champion_sigma = infer_full(
            champion, raw, norm, LEAD_INDEX[lead], rows, columns, args.batch_size, device
        )
        router_mu, router_sigma = infer_full(
            router, raw, norm, LEAD_INDEX[lead], rows, columns, args.batch_size, device
        )
        phase_selective_fusion(champion_mu, champion_sigma, router_mu, router_sigma)

    run_once()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        run_once()
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "status": "complete",
        "hardware": properties.name,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input_contract": {
            "members": 4,
            "variables": 6,
            "grid": [721, 1440],
            "lead_hours": lead,
            "tiling_patch": [96, 96],
            "batch_size": args.batch_size,
        },
        "parameters": {
            "global_skill_expert": parameter_count(champion),
            "phase_router_expert": parameter_count(router),
            "deployable_total": parameter_count(champion) + parameter_count(router),
            "selector_trainable": 0,
        },
        "warmup_runs": 1,
        "measured_runs": args.repeats,
        "full_system_seconds": durations,
        "mean_seconds": float(np.mean(durations)),
        "sample_std_seconds": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
        "median_seconds": float(np.median(durations)),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 2**30),
        "checkpoint_load_reports": {
            "global_skill_expert": champion_report,
            "phase_router_expert": router_report,
        },
        "scope": "sequential full global inference for both experts plus frozen selector; excludes disk input loading",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
