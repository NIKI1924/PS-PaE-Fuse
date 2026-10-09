# Frozen protocol for rapid 2022 weather-type stratification

Status: locked before weather-type scores were computed.

## Purpose and evidence boundary

This analysis reuses the same 32 fixed 2022 initialization dates and 72, 120 and 168 h leads used in the sealed focused evaluation. It does not add event-enriched dates, retrain a model, refit a router, change the safeguard, alter predictive-scale calibration, or select a threshold using 2022 comparative performance. It is a rapid process-stratified diagnostic of the sealed sample rather than an exhaustive annual event catalogue.

The event definitions are read without modification from `focused_event_definitions_2022.json`, whose definitions were frozen before the 2022 scores. Tropical-cyclone positions are taken from NOAA/NCEI IBTrACS v04r01. The downloaded 2022 subset is `ibtracs_2022_v04r01.csv`; its local SHA-256 is `657171B47FAB74DC3D92E36800221B870FE30483292509AAABF7246497124F9B`.

## Strata

1. **Tropical-cyclone neighbourhoods.** At each valid time, verification is restricted to grid cells within 500 km of an IBTrACS storm centre. Each unique IBTrACS SID is the independent resampling unit. Wind-event thresholds remain the frozen, pre-2022 gridwise q97.5 values. A storm contributes inferentially only if the verifying sample contains a non-empty q97.5 wind event in its neighbourhood.
2. **Cold-surge-associated wind.** For the 72--120 h and 120--168 h pairs, the final valid-time mask contains cells with an ERA5 2 m temperature fall of at least 6 K and a final temperature below the frozen pre-2022 local p05 threshold. The initialization date is the independent unit. Wind verification at the final lead is restricted to the affected mask and uses the frozen q97.5 wind threshold.
3. **Non-tropical high wind.** The verifying event is ERA5 wind at or above both the frozen local q97.5 threshold and 20 m s-1 after excluding all contemporaneous 500 km tropical-cyclone masks. The initialization date is the independent unit. Categorical and spatial scores use the non-tropical domain; conditional continuous scores use verifying event cells.

## Outputs and statistical rules

For Pangu-Weather, GraphCast, FuXi, HRES, the simple mean, EMOS, the phase router and the calibrated safeguard, the evaluator reports U10 RMSE, q97.5 CSI, POD, FAR and FSS9. CRPS and Brier score are reported only for probabilistic outputs. All uncertainty intervals and paired method differences resample the independent units specified above with 2,000 bootstrap replicates.

At least 10 independent units with a non-empty verifying event are required for inferential language. A smaller stratum is labelled descriptive only. Comparisons emphasize Pangu-Weather versus the calibrated safeguard for event detection and the phase router versus the calibrated safeguard for the deterministic--probabilistic operating-point trade-off.

Typical cases are selected using truth-only event area within each stratum, never the relative performance of either method. One largest truth-only case is retained for each stratum; these maps are descriptive and are not used for significance claims.

## Interpretation constraint

Because the primary population remains 32 sparsely sampled initialization dates, a positive result supports process consistency within the sealed sample but does not establish full-year climatological representativeness. The manuscript must state the number of independent systems or dates and must not describe the analysis as an exhaustive tropical-cyclone, cold-surge or windstorm catalogue.
