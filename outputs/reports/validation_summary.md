# CATEM Validation Summary

## Evidence Path

CATEM now follows the publishable research path:

Framework -> Data -> Validation -> Results -> Evidence

## Benchmark

- Dataset: `datasets/benchmark/catem_master.csv`
- Records: 3,540
- Target variable: `telepresence_quality`

## Regression Model Comparison

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Presence only | 0.018 | 0.149 | 0.208 |
| Workload only | 0.017 | 0.148 | 0.208 |
| System only | 0.076 | 0.134 | 0.202 |
| Performance only | 0.542 | 0.085 | 0.142 |
| CATEM full model | 0.587 | 0.082 | 0.135 |

## Current Result

The full CATEM model outperforms the single-layer regression baselines in this benchmark build.

This is the first evidence that CATEM can operate as a predictive validation framework, not only as a dashboard or conceptual taxonomy.
