# CATEM Dataset Strategy

CATEM does not require one perfect dataset. The better research strategy is to build a harmonized CATEM dataset from complementary sources.

## Combined Dataset Plan

```text
Dataset A: PhysioNet
  -> HR
  -> HRV
  -> EDA/GSR
  -> stress / physiology

Dataset B: ROS or robot telemetry logs
  -> latency
  -> motion quality
  -> navigation/task behavior
  -> tracking instability
  -> packet loss / jitter

Dataset C: NASA-TLX
  -> workload
  -> mental demand
  -> effort
  -> frustration

Dataset D: questionnaire or task outcome data
  -> overall telepresence quality
  -> user satisfaction
  -> task success
  -> immersion / presence outcome

Combined output:
  -> CATEM-compatible analysis table
```

## Why This Is Stronger

Most public datasets only cover one slice of telepresence. CATEM is useful because it can align partial modalities into one evaluation framework:

- physiology from wearable datasets
- workload from survey datasets
- system performance from robot/VR logs
- behavior from task logs
- outcome labels from questionnaires or performance metrics

This lets CATEM become a data harmonization and decision-support layer rather than a dashboard tied to one dataset.

## Minimum Viable CATEM Dataset

For each session, CATEM should ideally contain:

- `participant_id`
- `session_id`
- physiology: `heart_rate`, `hrv`, `gsr`
- workload: `nasa_tlx_score`, `mental_demand`, `effort`, `frustration`
- system: `latency_ms`, `jitter`, `packet_loss`, `tracking_loss`, `fps`
- behavior: `task_completion_time`, `error_rate`, `movement_smoothness`
- outcome: `overall_telepresence_quality`

## Current Implementation

The integration module supports this pipeline:

```python
from src.dataset_integrations import (
    combine_catem_sources,
    integrate_nasa_tlx_csv,
    integrate_physionet_wearable_csv,
    integrate_ros_telemetry_csv,
    save_integrated_dataset,
)

physiology = integrate_physionet_wearable_csv("data/external/physionet/session_01.csv", "P001", "S01")
telemetry = integrate_ros_telemetry_csv("data/external/ros/session_01.csv", "P001", "S01")
workload = integrate_nasa_tlx_csv("data/external/tlx/nasa_tlx.csv")

catem_dataset = combine_catem_sources(physiology, telemetry, workload)
save_integrated_dataset(catem_dataset)
```

## Next Research Step

Add a true outcome column such as `overall_telepresence_quality`, `user_satisfaction`, `task_success`, or `immersion_score`. This is what allows CATEM to be validated statistically against ground truth.
