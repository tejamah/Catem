# External Dataset Integration

CATEM supports local integration of public datasets into the CATEM schema. The project does not bundle external datasets because dataset terms, file sizes, and credential requirements differ by provider.

## Supported Sources

### PhysioNet

Source: https://physionet.org

Useful CATEM layers:

- Physiology
- Workload
- Data Quality

Expected local input:

- CSV exports from wearable, stress, ECG, EDA, HR, HRV, IBI, or physiology datasets

Adapter:

```python
from src.dataset_integrations import integrate_physionet_wearable_csv

df = integrate_physionet_wearable_csv(
    "data/external/physionet/example.csv",
    participant_id="P001",
    session_id="S01",
)
```

### OpenNeuro

Source: https://openneuro.org

Useful CATEM layers:

- Physiology
- Cognitive workload
- Behavior
- Task performance

Expected local input:

- Downloaded OpenNeuro/BIDS dataset folder
- Adapter scans for `*_events.tsv` files

Adapter:

```python
from src.dataset_integrations import integrate_openneuro_bids

df = integrate_openneuro_bids("data/external/openneuro/dsXXXXXX")
```

### ROS Telemetry Logs

Useful CATEM layers:

- System
- Behavior
- Task performance

Expected local input:

- CSV export from ROS/robot logs containing fields such as latency, FPS/rate, jitter, packet loss, tracking loss, task duration, command counts, path smoothness, or errors

Adapter:

```python
from src.dataset_integrations import integrate_ros_telemetry_csv

df = integrate_ros_telemetry_csv(
    "data/external/ros/session_01_telemetry.csv",
    participant_id="P001",
    session_id="S01",
)
```

### NASA-TLX

Useful CATEM layers:

- Workload

Expected local input:

- CSV survey export with participant/session identifiers and fields such as `nasa_tlx_score`, `mental_demand`, `effort`, or `frustration`

Adapter:

```python
from src.dataset_integrations import integrate_nasa_tlx_csv

df = integrate_nasa_tlx_csv("data/external/tlx/nasa_tlx.csv")
```

### DEAP

Source: DEAP dataset host / licensing flow

Useful CATEM layers:

- Physiology
- Stress
- Engagement
- Workload proxy

Expected local input:

- DEAP preprocessed participant `.dat` files

Adapter:

```python
from src.dataset_integrations import integrate_deap_preprocessed_dat

df = integrate_deap_preprocessed_dat("data/external/deap/s01.dat")
```

## CATEM Output

All adapters return a CATEM-compatible DataFrame with the columns expected by the scoring engine:

- participant/session/timestamp identifiers
- embodiment metrics
- presence metrics
- behavior metrics
- physiology metrics
- workload metrics
- system metrics
- data-quality metrics
- optional `overall_telepresence_quality`

## Suggested Folder Layout

```text
data/
  external/
    physionet/
    ros/
    tlx/
    openneuro/
    deap/
  processed/
    catem_external_integrated.csv
```

## Saving Integrated Data

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
save_integrated_dataset(catem_dataset, "data/processed/catem_external_integrated.csv")
```

## Important Limitation

Public physiology and neuroscience datasets rarely contain every CATEM layer. For example, DEAP is strong for physiology and affective engagement, but it does not contain teleoperation latency or task telemetry. These datasets should therefore be used as partial validation sources until Cornell or experiment-specific telepresence data is available.

The recommended CATEM strategy is to combine complementary datasets instead of searching for one perfect dataset:

- PhysioNet for HR, HRV, EDA/GSR, stress, and physiology
- ROS/robot logs for telemetry, navigation, motion, and task behavior
- NASA-TLX for workload
- questionnaire or task outcome data for ground-truth telepresence quality
