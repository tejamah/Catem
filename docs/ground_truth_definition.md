# Ground Truth Definition

## Target Variable

The current benchmark target is `telepresence_quality`.

This field is a proxy target, not yet a true ground-truth score from a controlled telepresence user study.

## How It Is Created

CATEM Benchmark v1 merges several public or user-provided sources. Because these datasets do not share one common telepresence outcome variable, `telepresence_quality` is mapped from the best available outcome-like signal in each source:

| Source | Proxy Used |
| --- | --- |
| NASA-TLX files | Task performance score from the workload study export |
| RoboTurk metrics | Visual prediction quality from PSNR/SSIM-derived success proxies |
| Zenodo multimodal workbook | Performance, situation-awareness, and trust proxies |
| Robot anomaly dataset | Non-anomaly rate, scan quality, motion stability, battery state, and latency quality |
| Apple Health export | Daily activity, walking quality, sleep/recovery, and low workload-risk proxy |

## Why This Is Useful

This proxy target is useful for testing whether the ETL, scoring, validation, dashboard, and publication pipeline work end to end.

It supports a benchmark prototype claim:

CATEM can integrate multimodal datasets and compare cross-layer models against single-layer baselines.

## What It Does Not Prove Yet

It does not yet prove that CATEM predicts true telepresence quality in real users.

The current target is partly derived from variables that are also available to CATEM models. That means the current R2 values should be interpreted as pipeline validation and proxy benchmark evidence, not final scientific proof.

## Required Next Step

For a defensible research paper, collect or obtain a real telepresence study with an independent outcome variable such as:

- post-session telepresence quality rating
- validated presence or embodiment questionnaire total
- user satisfaction score
- task success judged independently by an evaluator
- expert-rated operator performance

Then rerun the same validation scripts with `telepresence_quality` set to that independent outcome.
