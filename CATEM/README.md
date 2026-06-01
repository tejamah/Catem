# CATEM Dataset v1

This folder is the research benchmark artifact for CATEM.

## Folder Structure

```text
CATEM/
  datasets/
    physionet/
    nasa_tlx/
    roboturk/
    rosbags/
    cisc/
  benchmark/
    catem_v1.csv
  scripts/
    build_catem_v1.py
    run_validation.py
  outputs/
    catem_v1_long.csv
    catem_v1_validation.csv
    catem_v1_correlations.csv
    catem_v1_regression.csv
    catem_v1_feature_importance.csv
```

## Master CSV

`benchmark/catem_v1.csv` uses the unified CATEM schema:

```text
participant_id, session_id, task_id, timestamp,
heart_rate, hrv, ecg_signal, gsr, eda, skin_temp,
mental_demand, physical_demand, temporal_demand, effort, frustration, nasa_tlx_total,
latency_ms, tracking_error, packet_loss, jitter, fps,
task_completion_time, error_rate, success_rate, path_efficiency,
ownership_score, agency_score, presence_score,
catem_score
```

The generated file also keeps intermediate validation columns such as `overall_telepresence_quality`, `physiology_score`, `workload_score`, `system_score`, and `performance_score`.

## Build Command

From the repository root:

```bash
python CATEM/scripts/build_catem_v1.py
```

## First Research Goal

Build CATEM Dataset v1 with 1000+ records, then run:

- correlation analysis
- feature importance
- regression models
- random forest
- CATEM validation

Current local build uses the downloaded NASA-TLX, RoboTurk, Zenodo workbook, and robot anomaly files available in `Downloads`.

## Validation Command

```bash
python CATEM/scripts/run_validation.py
```

This produces:

- `outputs/catem_v1_correlations.csv`
- `outputs/catem_v1_regression.csv`
- `outputs/catem_v1_feature_importance.csv`
