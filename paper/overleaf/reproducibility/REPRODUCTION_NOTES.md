# Reproduction notes

1. Obtain upstream Pangu-Weather, GraphCast, FuXi and HRES forecasts and ERA5
   verification data under the original providers' terms.
2. Harmonize every field to the 0.25-degree latitude--longitude grid, exact
   valid time, common units and north-to-south latitude order.
3. Use the 2020 development data only to fit thresholds, the shared Gaussian
   scale and any EMOS parameters.
4. Download the four frozen PS-PaE-Fuse assets from the published GitHub Release
   `v1.0-paper-assets` and verify their SHA-256 hashes against
   `checkpoints/manifest.json`. All four remote digests were verified at publication.
5. Run the cross-year and 2022 evaluations from `evaluation/`, replacing the
   historical site-specific data roots with local paths.
6. Build the manuscript with `pdflatex main.tex` twice from `paper/overleaf/`.

The scripts under `archive/server_scripts/` preserve the original server paths
and are included as execution provenance. They are not claimed to be portable
launchers without environment- and path-specific editing.

Paths in these instructions refer to the GitHub repository layout. This Overleaf
package includes a copy of the checkpoint manifest as
`reproducibility/checkpoint_manifest.json`; it does not bundle model binaries.

