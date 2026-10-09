# PS-PaE-Fuse

**PS-PaE-Fuse** expands to **Phase-Selective Phase-Aware Ensemble Fusion**.

This repository contains the code, frozen protocols, evaluation scripts, figure
source and AAS manuscript source associated with:

> *Extreme-Aware Fusion of Multi-Model AI Weather Forecasts: Joint Optimization
> of Overall Skill and Event Detection*

PS-PaE-Fuse post-processes Pangu-Weather, GraphCast, FuXi and ECMWF HRES on a
common 0.25-degree grid. It produces deterministic fields, Gaussian scale
parameters and event probabilities for six variables at 72, 120 and 168 h. A
forecast-only selector blends a global-skill expert with a phase-aware selective
expert, and a frozen consensus safeguard sets the operating point for extreme
surface-wind detection.

## Repository map

- `src/`: reusable fusion modules.
- `training/`: local training entry points.
- `evaluation/`: cross-year, probabilistic, FSS, weather-type and runtime
  evaluation scripts.
- `protocols/`: frozen 2022 event definitions and evaluation protocols.
- `archive/server_scripts/`: exact historical training and upstream-model
  inference scripts used on the authors' server. Paths in these files are
  site-specific provenance, not portable defaults.
- `figures/multivariate_probability_2021/`: source data, code and vector/raster
  output for the six-variable skill and probabilistic-verification figure.
- `paper/overleaf/`: Overleaf-ready manuscript and supplementary source.
- `docs/`: model provenance, training history and reproduction notes.
- `checkpoints/manifest.json`: filenames, sizes and SHA-256 hashes for the
  frozen checkpoints.

## Frozen checkpoints

The four frozen checkpoint files are being deposited as assets of the
[`v1.0-paper-assets`](https://github.com/NIKI1924/PS-PaE-Fuse/releases/tag/v1.0-paper-assets)
release. They are not stored in Git history because each file is approximately
117 MB. Publication is pending until all four assets appear on the release.
Verify every download against `checkpoints/manifest.json`.

## Latest manuscript

The Overleaf sources include the author-confirmed Figure 6 layout and author
metadata: Ruxue XING and Jianjun ZHU share first authorship; all four authors
are affiliated with China Agricultural University; Yaojun WANG is the
corresponding author. The title has not changed.

Compile `main.tex` twice with pdfLaTeX from `paper/overleaf/`, or compile
`supplement.tex` separately for the supplementary material. Release assets
will include the complete Overleaf ZIP and compiled PDFs for this version.

## Upstream forecasts and data

Pangu-Weather, HRES and ERA5 fields were read from WeatherBench 2. The 2021--2022
GraphCast and FuXi fields were generated locally from released checkpoints; see
[`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md) for exact checkpoint
identities, hashes, initialization sources and rollout settings. Raw upstream
forecast archives are not redistributed here and remain subject to their
original providers' terms.

## Reproducing the new combined figure

```bash
python figures/multivariate_probability_2021/plot_multivariate_probability_2021.py
```

The script reads only the two files in the adjacent `data/` directory and writes
PDF, SVG, 300 dpi PNG and a manifest to `output/`.

## Environment

The fusion and evaluation code was run with Python 3.10 and PyTorch. A compact
dependency list is provided in `requirements.txt`. GraphCast and FuXi generation
additionally require the environments and released model packages described in
their upstream repositories.

## Scope and licensing

This is a research artifact. No software license has yet been granted; all
rights are reserved unless a license is added by the repository owner. Upstream
models, datasets and third-party code retain their own licenses and terms.

