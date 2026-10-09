#!/usr/bin/env python3
"""Reproduce the six-variable skill and wind-reliability composite.

The script reads only the archived 2021 evaluation records shipped beside it.
No reported score is hard-coded in the plotting code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


VARIABLES = ("T2M", "U10", "V10", "MSL", "Z500", "T850")
LEADS = ("72", "120", "168")
METHODS = (
    ("GraphCast", "GraphCast"),
    ("FuXi", "FuXi"),
    ("HRES corrected exact lead", "HRES"),
    ("Simple arithmetic mean", "Simple mean"),
    ("Global-skill fusion", "Global-skill expert"),
    (
        "Frozen phase-selective fusion mean of three seed metrics",
        "PS-PaE-Fuse\nwithout safeguard",
    ),
)
ROUTER_METHODS = (
    "phase_selective_router_s123",
    "phase_selective_router_s456",
    "phase_selective_router_s789",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_skill(path: Path) -> dict[tuple[str, str, str], float]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["stratum"] == "All grid cells"]
    values = {
        (row["method"], row["variable"], row["lead"]): float(row["rmse"])
        for row in selected
    }
    required_methods = {"Pangu", *(method for method, _ in METHODS)}
    for method in required_methods:
        for variable in VARIABLES:
            for lead in LEADS:
                if (method, variable, lead) not in values:
                    raise KeyError(f"Missing RMSE: {method}, {variable}, {lead} h")
    return values


def skill_matrix(
    values: dict[tuple[str, str, str], float], lead: str
) -> np.ndarray:
    matrix = np.empty((len(METHODS), len(VARIABLES)), dtype=float)
    for row, (method, _label) in enumerate(METHODS):
        for column, variable in enumerate(VARIABLES):
            baseline = values[("Pangu", variable, lead)]
            candidate = values[(method, variable, lead)]
            matrix[row, column] = 100.0 * (baseline - candidate) / baseline
    return matrix


def aggregate_router(metrics: dict, lead: str) -> dict[str, np.ndarray | float]:
    records = [metrics[name][lead] for name in ROUTER_METHODS]
    weights = np.asarray(
        [record["reliability"]["wind_q975"]["weight"] for record in records],
        dtype=float,
    )
    forecast = np.asarray(
        [
            record["reliability"]["wind_q975"]["mean_forecast_probability"]
            for record in records
        ],
        dtype=float,
    )
    observed = np.asarray(
        [
            record["reliability"]["wind_q975"]["observed_frequency"]
            for record in records
        ],
        dtype=float,
    )
    total_weight = weights.sum(axis=0)
    return {
        "forecast": (weights * forecast).sum(axis=0) / total_weight,
        "observed": (weights * observed).sum(axis=0) / total_weight,
        "weight": weights.mean(axis=0),
        "brier": float(
            np.mean([record["Brier"]["wind_q975"]["estimate"] for record in records])
        ),
        "crps": float(
            np.mean([record["traditional"]["U10_CRPS"]["estimate"] for record in records])
        ),
    }


def extract_emos(metrics: dict, lead: str) -> dict[str, np.ndarray | float]:
    record = metrics["EMOS"][lead]
    reliability = record["reliability"]["wind_q975"]
    return {
        "forecast": np.asarray(reliability["mean_forecast_probability"], dtype=float),
        "observed": np.asarray(reliability["observed_frequency"], dtype=float),
        "weight": np.asarray(reliability["weight"], dtype=float),
        "brier": float(record["Brier"]["wind_q975"]["estimate"]),
        "crps": float(record["traditional"]["U10_CRPS"]["estimate"]),
    }


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw(
    skill_csv: Path,
    probability_json: Path,
    output_stem: Path,
) -> None:
    set_style()
    skill_values = load_skill(skill_csv)
    payload = json.loads(probability_json.read_text(encoding="utf-8"))
    metrics = payload["metrics"]

    negative = "#B34A32"
    neutral = "#F6F5F1"
    positive = "#087E8B"
    cmap = LinearSegmentedColormap.from_list(
        "skill_diverging", [negative, "#EBC5B8", neutral, "#A9D7D3", positive]
    )
    norm = TwoSlopeNorm(vmin=-35.0, vcenter=0.0, vmax=22.0)
    ours_color = "#6A3D9A"
    emos_color = "#2B7BBB"

    # A separate footer band is reserved for scores and a common legend.
    # Nothing is placed over reliability curves or below an individual xlabel.
    fig = plt.figure(figsize=(8.6, 7.6), constrained_layout=False)
    left, right, gap = 0.20, 0.965, 0.028
    panel_width = (right - left - 2 * gap) / 3
    column_lefts = [left + column * (panel_width + gap) for column in range(3)]

    heatmaps = []
    heatmap_axes = []
    for column, lead in enumerate(LEADS):
        ax = fig.add_axes([column_lefts[column], 0.710, panel_width, 0.235])
        heatmap_axes.append(ax)
        matrix = skill_matrix(skill_values, lead)
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        heatmaps.append(image)
        # Lead time is a shared column heading, not a chart title.
        fig.text(
            column_lefts[column] + panel_width / 2, 0.957, f"{lead} h",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        ax.set_xticks(range(len(VARIABLES)), VARIABLES, rotation=45, ha="right")
        ax.set_yticks(range(len(METHODS)))
        if column == 0:
            ax.set_yticklabels([label for _method, label in METHODS])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-0.5, len(VARIABLES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.15)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0)
        for row in range(matrix.shape[0]):
            for variable_index in range(matrix.shape[1]):
                value = matrix[row, variable_index]
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if luminance < 0.52 else "#202020"
                ax.text(
                    variable_index,
                    row,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=color,
                    fontweight="bold" if row == len(METHODS) - 1 else "normal",
                )
        ax.text(
            -0.13,
            1.055,
            chr(ord("a") + column),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if column == 0:
            ax.set_ylabel("Forecast system", labelpad=8)

    fig.canvas.draw()
    first_position = heatmap_axes[0].get_position()
    last_position = heatmap_axes[-1].get_position()
    colorbar_ax = fig.add_axes(
        [first_position.x0, 0.635, last_position.x1 - first_position.x0, 0.013]
    )
    colorbar = fig.colorbar(heatmaps[0], cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label("RMSE skill relative to Pangu-Weather (%)  |  higher is better", labelpad=5)
    colorbar.set_ticks([-35, -20, -10, 0, 10, 20])

    legend_handles = None
    for column, lead in enumerate(LEADS):
        square_height = panel_width * 8.6 / 7.6
        ax = fig.add_axes([column_lefts[column], 0.305, panel_width, square_height])
        sharpness_ax = fig.add_axes(
            [column_lefts[column], 0.219, panel_width, 0.065], sharex=ax
        )
        score_ax = fig.add_axes([column_lefts[column], 0.080, panel_width, 0.066])
        ours = aggregate_router(metrics, lead)
        emos = extract_emos(metrics, lead)

        ax.plot([0, 1], [0, 1], color="#8D8D8D", linewidth=0.9, linestyle="--", zorder=1)
        for record, color, label, marker in (
            (emos, emos_color, "EMOS", "o"),
            (ours, ours_color, "PS-PaE-Fuse without safeguard", "D"),
        ):
            forecast = np.asarray(record["forecast"])
            observed = np.asarray(record["observed"])
            weight = np.asarray(record["weight"])
            sizes = 15.0 + 58.0 * np.sqrt(weight / weight.max())
            line, = ax.plot(
                forecast,
                observed,
                color=color,
                linewidth=1.35,
                marker=marker,
                markersize=3.6,
                markerfacecolor="white",
                markeredgewidth=0.9,
                zorder=3,
                label=label,
            )
            ax.scatter(
                forecast,
                observed,
                s=sizes,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.24,
                zorder=2,
            )
            if legend_handles is None:
                legend_handles = []
            if column == 0:
                legend_handles.append(line)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#D9DEE2", linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)
        if column == 0:
            ax.set_ylabel("Observed frequency")
        else:
            ax.set_yticklabels([])
        # Scores occupy a dedicated, aligned strip outside the data area.
        score_ax.set_xlim(0, 1)
        score_ax.set_ylim(0, 1)
        score_ax.axis("off")
        for x, text_value, align in (
            (0.0, "Method", "left"),
            (0.69, "Brier", "right"),
            (1.0, "U10 CRPS", "right"),
        ):
            score_ax.text(x, 0.99, text_value, ha=align, va="top", fontsize=8.0)
        score_ax.axhline(0.67, color="#C8CDD1", linewidth=0.65)
        for y, record, label, color in (
            (0.49, ours, "PS-PaE-Fuse*", ours_color),
            (0.10, emos, "EMOS", emos_color),
        ):
            score_ax.text(0.0, y, label, color=color, ha="left", va="center", fontsize=8.0)
            score_ax.text(0.69, y, f"{record['brier']:.4f}", ha="right", va="center", fontsize=8.0)
            score_ax.text(1.0, y, f"{record['crps']:.3f}", ha="right", va="center", fontsize=8.0)
        ax.text(
            0.025,
            0.97,
            chr(ord("d") + column),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        centers = np.arange(0.05, 1.0, 0.1)
        for record, color, offset in ((emos, emos_color, -0.014), (ours, ours_color, 0.014)):
            weight = np.asarray(record["weight"])
            share = weight / weight.sum()
            sharpness_ax.bar(
                centers + offset,
                share,
                width=0.026,
                color=color,
                alpha=0.80,
                edgecolor="none",
            )
        sharpness_ax.set_yscale("log")
        sharpness_ax.set_ylim(1e-5, 1.1)
        sharpness_ax.set_yticks([1e-4, 1e-2, 1])
        sharpness_ax.set_yticklabels([".01%", "1%", "100%"] if column == 0 else [])
        sharpness_ax.grid(axis="y", color="#E3E6E8", linewidth=0.5)
        sharpness_ax.spines[["top", "right"]].set_visible(False)
        sharpness_ax.tick_params(axis="x", pad=3)
        sharpness_ax.tick_params(axis="y", pad=3)
        if column == 0:
            sharpness_ax.set_ylabel("Bin share", labelpad=4)

    fig.text(
        (left + right) / 2, 0.174, "Forecast probability",
        ha="center", va="center", fontsize=9.0,
    )
    fig.text(
        left, 0.027, "* Without safeguard  |  Wind event: local q97.5  |  U10 CRPS in m s$^{-1}$",
        ha="left", va="center", fontsize=8.0, color="#333333",
    )

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=((left + right) / 2, 0.042),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=2.2,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": output_stem.name,
        "scope": "Corrected 2021 all-grid six-variable RMSE and q97.5 wind reliability",
        "skill_definition": "100 * (RMSE_Pangu - RMSE_method) / RMSE_Pangu",
        "probability_comparison": "three-seed mean of phase-selective router output without safeguard versus EMOS",
        "input_files": {
            skill_csv.name: sha256(skill_csv),
            probability_json.name: sha256(probability_json),
        },
        "lead_hours": [72, 120, 168],
        "variables": list(VARIABLES),
        "methods": [label.replace("\n", " ") for _method, label in METHODS],
        "outputs": {
            suffix: sha256(output_stem.with_suffix(suffix))
            for suffix in (".pdf", ".svg", ".png")
        },
    }
    output_stem.with_name(output_stem.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--skill-csv", type=Path, default=root / "data" / "all_metrics.csv")
    parser.add_argument(
        "--probability-json",
        type=Path,
        default=root / "data" / "spatial_probabilistic_2021_extended.json",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=root / "output" / "fig_multivariate_probability_2021",
    )
    args = parser.parse_args()
    draw(args.skill_csv, args.probability_json, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
