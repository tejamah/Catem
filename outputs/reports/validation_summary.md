# CATEM Validation Summary

## Evidence Path

CATEM now follows the publishable research path:

Framework -> Data -> Validation -> Results -> Evidence

## Benchmark

- Dataset: `datasets/benchmark/catem_master.csv`
- Records: 4,524
- Target variable: `telepresence_quality`
- New wearable source: Apple Health daily activity and sleep proxy records

## Regression Model Comparison

| Model | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Presence only | 0.017 | 0.131 | 0.188 |
| Workload only | 0.016 | 0.130 | 0.188 |
| System only | 0.072 | 0.120 | 0.182 |
| Performance only | 0.643 | 0.075 | 0.113 |
| CATEM full model | 0.711 | 0.070 | 0.102 |

## Current Result

The full CATEM model outperforms the single-layer regression baselines in this benchmark build.

This is the first evidence that CATEM can operate as a predictive validation framework, not only as a dashboard or conceptual taxonomy.
