# Build and quality-control record 20261009

## Public-release update v12

All four frozen checkpoints were uploaded to GitHub Release
`v1.0-paper-assets`; GitHub returned SHA-256 digests matching the local manifest
for every checkpoint. The manuscript availability statement was changed from
future deposition to publicly available. Author metadata, the title, Figure 6
and scientific results are unchanged. Earlier sections below preserve the
historical build states and should not be interpreted as the current upload
status. The remaining author declarations still require confirmation.

## Author metadata update v11

Both main and supplementary author blocks were updated from the authors'
supplied screenshots and corrections: Ruxue XING and Jianjun ZHU share the
equal-contribution dagger; Lang ZHENG and all other authors use the same China
Agricultural University affiliation; Yaojun WANG is the corresponding author,
with email `wangyaojun@cau.edu.cn`. The institution is College of Information
and Electrical Engineering, China Agricultural University, Beijing 100083,
China. The old author/affiliation placeholders were removed.

Both documents compiled twice. Their first pages were rendered and visually
checked with no observed overlaps. Main: 43 pages; supplement: 17 pages.
The title, Figure 6, scientific body text, all tables and other figures remain
unchanged from v10. Remaining author contribution/funding declarations were
not invented. No GitHub upload was performed.

## Previous author-approved Figure 6 insertion v10

This version inserts the author-approved standalone layout into main Fig. 6.
The revised plotting script differs from the approved preview script only in
its default output basename. The inserted PNG SHA-256 equals the approved
preview exactly: `a306c49c9a84177de2893950182d43f51b569f47d108ff665c868221426a8129`.
The PDF was regenerated, so its binary metadata/checksum is different.

Main Fig. 6 is now displayed at full text width. Only this figure's caption
line spacing was adjusted, and its reference to score insets was changed to
aligned score strips. The main PDF was compiled twice and its Fig. 6 page 25
was rasterized and visually inspected: figure, labels and complete caption
are visible, with no observed overlap or clipping. No new oversized-float
warning was introduced for Fig. 6; inherited warnings are unchanged.

All other nine figure files, all 14 table sources, the supplement source and
PDF, the title and scientific body text match the previous version. No GitHub
upload was performed. The updated main PDF remains 43 pages and its fonts are
embedded; final compilation has no undefined references or missing characters.

## Scope

This delivery updates the AAS manuscript locally. It does not change the agreed title or Figure 1 artwork. Further GitHub and binary-checkpoint uploads were deferred at the author's request.

## Clean build

- Main file: `main.tex`; supplementary file: `supplement.tex`.
- Compiler: pdfLaTeX, MiKTeX 25.12; each file compiled twice from a clean package directory.
- Main PDF: 43 pages. Supplementary PDF: 17 pages.
- Final logs contain no undefined references, undefined citations or missing-character warnings.
- All PDF fonts are embedded and extractable: 30 unique font objects in the main PDF and 20 in the supplement, checked with PyMuPDF.
- The repository and planned release hyperlinks are present on main-PDF page 34.

## Visual inspection

The new six-panel figure on page 25, availability paragraphs on pages 33–34, and existing large-figure pages 19, 22 and 28 were rendered and visually inspected. No clipping or text overlap was observed on these inspected pages. This is a targeted inspection, not a claim of independent scientific revalidation of every result or of every page.

Inherited template warnings remain: fancyhdr head height, perpage placement, an absent footnote destination, and three old floats exceeding the nominal text height. Their corresponding pages were inspected and their figures/captions remain visible. The persistent label-change warning is not accompanied by undefined references.

## Scientific scope of the added figure

The new main Fig. 6 uses existing 2021 evaluation outputs. Its PS-PaE-Fuse deterministic and probability values are explicitly pre-safeguard; they are not relabeled as the final safeguarded system. The figure combines three six-variable RMSE-skill heatmaps, reliability curves, probability-bin populations, and Brier/U10-CRPS comparisons. Original aggregate input data, plotting code, vector PDF/SVG and 300-dpi PNG are included under `figure_sources/fig_multivariate_probability_2021/`.

## Provenance and pending items

GraphCast/FuXi checkpoint versions, initialization archives, cascade/rollout settings and full checksums are in `MODEL_PROVENANCE.md`. The global-skill launch and transcribed 27-epoch history are in `GLOBAL_SKILL_TRAINING_RECORD.md` and `global_skill_training_history.csv`.

Four trained checkpoint binaries have been downloaded and hash-verified locally, but are not included in this Overleaf ZIP and are not yet downloadable from the planned GitHub release. The manuscript therefore states that they will be deposited before final submission. The checksums are in `checkpoint_manifest.json`.

Author names, equal first authorship, the common affiliation and correspondence were subsequently supplied by the authors and filled in the author-confirmed version. Funding, author contributions, competing-interest confirmation and the LLM-use declaration still require the authors' own entries. The remaining yellow placeholders are retained rather than invented.
