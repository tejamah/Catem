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
    openneuro/
    deap/
  processed/
    catem_external_integrated.csv
```

## Saving Integrated Data

```python
from src.dataset_integrations import save_integrated_dataset

save_integrated_dataset(df, "data/processed/catem_external_integrated.csv")
```

## Important Limitation

Public physiology and neuroscience datasets rarely contain every CATEM layer. For example, DEAP is strong for physiology and affective engagement, but it does not contain teleoperation latency or task telemetry. These datasets should therefore be used as partial validation sources until Cornell or experiment-specific telepresence data is available.
