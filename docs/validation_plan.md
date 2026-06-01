# Validation Plan

## Goal

Test whether CATEM explains telepresence quality better than single metrics.

## Comparison targets

- Presence only
- Embodiment only
- Workload only
- Performance only
- System telemetry only
- CATEM combined score

## Proposed methods

- Correlation analysis
- Linear regression
- Random forest feature importance
- Prediction accuracy

## Steps

1. Load synthetic dataset and compute CATEM layer scores.
2. Compare single-layer predictors with the combined CATEM score.
3. Use regression and random forest models to measure explanatory power.
4. Report validation metrics and feature importance.
