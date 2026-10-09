# PS-PaE-Fuse focused 2022 frozen validation protocol

Protocol date: 2026-09-13  
Target journal: *npj Climate and Atmospheric Science*  
Status: frozen before any 2022 PS-PaE-Fuse/Pangu comparative score is computed

## Scope

This extension deliberately prioritizes one auditable validation over many
experiments. It uses the intersection of already committed 2022 GraphCast and
FuXi Model-A initializations (32 dates sampled every 11 days from January to
December), three lead times (72, 120 and 168 h), and the already trained router
seeds 123, 456 and 789. No neural network is retrained.

## Frozen systems

- Pangu-Weather
- GraphCast
- FuXi
- exact-lead ECMWF HRES
- simple four-member mean
- 2020-fitted EMOS where the existing fit is directly reusable
- frozen phase-router-only PS-PaE-Fuse, seeds 123/456/789
- the already selected forecast-only consensus safeguard
- one low-parameter conditional scale calibration fitted exclusively with
  blocked out-of-fold 2020 data

The safeguard remains: locally strongest member, local 2020 q95 trigger,
minimum two-member support, alpha 1.0, zero additional speed margin. It may not
be changed after this protocol is frozen.

## Primary endpoints

1. cosine-latitude-weighted U10 RMSE;
2. q97.5 wind CSI;
3. q97.5 wind FSS with a 9-grid-cell neighbourhood;
4. q97.5 wind Brier score and reliability;
5. U10 Gaussian CRPS.

Secondary endpoints are q95 CSI/FSS/Brier and deterministic 15, 20 and
25 m s-1 CSI, POD, FAR, FBIAS and SEDI. Results are aggregated annually and by
DJF/MAM/JJA/SON. Uncertainty uses initialization-date block bootstrap; model
seeds are retained as repeated training realizations rather than treated as
independent weather cases.

## Conditional scale calibration

The safeguarded mean is fixed. Only its U10/V10 Gaussian scale may be adjusted
by a small forecast-only model using lead time, safeguard gate, mean shift and
raw-member spread. Coefficients are selected by blocked 2020 cross-validation,
then refitted on all 2020 data and serialized before 2022 scores are opened.
ERA5 event occurrence is never an inference input. Calibration is accepted for
the frozen test regardless of whether its 2022 result improves.

## Event stratification

No event-enriched dates are added in the primary run. The same 32 dates are
stratified by definitions fixed independently of model performance:

- tropical-cyclone influence: IBTrACS-valid-time centres and fixed-radius masks;
- cold surge: an ERA5 truth-defined 48 h temperature-drop tail combined with a
  locally cold final-state tail, with climatological thresholds fixed outside
  2022;
- non-tropical high wind: ERA5 q97.5/absolute-threshold connected objects after
  removal of tropical-cyclone masks.

If a stratum contains fewer than 10 independent systems, it is labelled
descriptive and no superiority claim is made. No extra case is downloaded merely
because a model looks favourable on it.

## 25 m s-1 attribution

The 2021 set may be used only to decompose misses, false alarms, intensity bias,
object displacement, gate support and selected-member source. It may not select
any new hyperparameter. Any repair candidate must be selected entirely in 2020
before the 2022 comparison is opened; otherwise it is deferred to a later-year
test.

## Test-seal rule

Data presence, units, shapes, valid times, checksums and missing values may be
audited before the seal is opened. Comparative 2022 skill tables are generated
only after the calibration JSON, event-definition JSON, evaluator code and this
protocol have been hashed. Once any 2022 comparative result has been inspected,
no model, threshold, calibration coefficient, event definition or date list may
be changed and still call 2022 an untouched test.

