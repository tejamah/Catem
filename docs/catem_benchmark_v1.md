# CATEM Benchmark Dataset v1

CATEM Benchmark Dataset v1 is a unified CSV schema for validating CATEM across partial public datasets. The goal is not to find one perfect dataset; the goal is to harmonize complementary sources into one benchmark table.

## Public Source Plan

| CATEM Layer | Source | What It Gives |
| --- | --- | --- |
| Physiology | PhysioNet wearable stress datasets | BVP/heart activity, HR/HRV, EDA/GSR, temperature, motion during stress tasks |
| Workload | UTA4 / UTA7 NASA-TLX datasets or local NASA-TLX exports | Mental demand, effort, frustration, workload scores |
| Robot/Telemetry | RoboTurk or robot teleoperation logs | Remote manipulation behavior, task performance, operator interaction |
| System/ROS | ROS bag/CSV exports | Robot telemetry, sensor streams, navigation logs, latency, tracking, packet loss |
| Human-state rich data | CISC-LIVE-LAB-3-style multimodal studies | NASA-TLX, eye tracking, EEG, health watch, stress, performance data |

## Master CSV Schema

```text
participant_id,session_id,task_id,timestamp,
heart_rate,hrv,ecg_signal,gsr,eda,skin_temp,
mental_demand,physical_demand,temporal_demand,effort,frustration,nasa_tlx_total,
latency_ms,tracking_error,packet_loss,jitter,fps,
task_completion_time,error_rate,success_rate,path_efficiency,
ownership_score,agency_score,presence_score,
overall_telepresence_quality,
physiology_score,workload_score,system_score,performance_score,
catem_score
```

## Stage 1: Collect Public Datasets

Download or export local files into:

```text
data/external/
  physionet/
  tlx/
  robot/
  ros/
  cisc_live_lab/
```

The repository ignores `data/external/` so large downloaded datasets are not committed.

## Stage 2: Normalize Each Source

Every source can be converted to a long traceability table:

```text
participant_id
session_id
task_id
timestamp
metric_name
metric_value
source_dataset
```

The builder writes this to:

```text
data/processed/catem_benchmark_v1_long.csv
```

## Stage 3: Create CATEM Layer Scores

The benchmark builder computes:

- Physiology Score = HRV + GSR/EDA + ECG/heart activity features
- Workload Score = NASA-TLX total and subscales
- System Score = latency + tracking + packet loss + jitter + FPS
- Performance Score = task time + error rate + success rate + path efficiency

## Stage 4: Compute CATEM Score

Initial benchmark formula:

```text
CATEM =
Embodiment
+ Presence
+ Performance
+ Physiology
+ System
- Workload
```

Later versions can learn weights using regression or random forest.

## Stage 5: Validate

The builder compares:

- Presence only
- Workload only
- Performance only
- System only
- Physiology only
- CATEM full model

Outputs:

```text
outputs/catem_benchmark_validation.csv
```

Validation goal:

Show that CATEM predicts `overall_telepresence_quality` better than any single metric group.

## Build Command

```bash
python scripts/build_catem_benchmark.py
```

Generated files:

```text
data/processed/catem_benchmark_v1.csv
data/processed/catem_benchmark_v1_long.csv
outputs/catem_benchmark_validation.csv
```

## Build From Provided Local Files

If the NASA-TLX and RoboTurk files are in `Downloads`, run:

```bash
python scripts/build_user_catem_benchmark.py
```

Expected inputs:

```text
Downloads/Demographics.csv
Downloads/nasatlx_assistant.csv
Downloads/nasatlx_current.csv
Downloads/roboturk_real_dataset-master/roboturk_real_dataset-master/results/
  psnr_bair_action.csv
  psnr_laundry_layout.csv
  psnr_tower_creation.csv
  ssim_bair_action.csv
  ssim_laundry_layout.csv
  ssim_tower_creation.csv
Downloads/dataset-main/concatenated_data_zenodo.xlsx
Downloads/robot_anomaly_full.csv
```

Generated files:

```text
data/processed/catem_benchmark_user_data.csv
data/processed/catem_benchmark_user_data_long.csv
outputs/catem_benchmark_user_validation.csv
```

Mapping:

- NASA-TLX assistant/current files -> workload and task performance proxy
- RoboTurk PSNR/SSIM curves -> system/performance quality proxy
- Zenodo workbook -> NASA-TLX, SART/SPAM, simulator logs, watch EDA/pulse/temperature, AI-support measures
- Robot anomaly CSV -> latency, message rate, scan validity, anomaly/error rate, motion stability, battery/system health
- Demographics file -> inspected for participant metadata, but not added to the master benchmark schema unless a study-specific metadata table is needed

## Current Status

The code path is complete. Until the external datasets are downloaded into `data/external/`, the build command uses the project synthetic CATEM sample as a working benchmark seed so the schema, layer scoring, and validation pipeline are executable end to end.
