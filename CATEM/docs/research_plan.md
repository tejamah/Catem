# CATEM Research Plan

## Phase 1: Benchmark Dataset v1

Tasks:

- Download or stage PhysioNet WESAD, NASA-TLX, RoboTurk, ROS datasets, and CISC-LIVE-LAB-3.
- Create CATEM folder structure.
- Create the unified data dictionary.

Output:

- `CATEM/benchmark/catem_v1.csv`

## Phase 2: ETL Pipeline

Scripts:

- `CATEM/scripts/load_physionet.py`
- `CATEM/scripts/load_nasatlx.py`
- `CATEM/scripts/load_roboturk.py`
- `CATEM/scripts/load_ros.py`
- `CATEM/scripts/merge_catem.py`

Target:

- 1000+ records after additional source downloads are added.

## Phase 3: CATEM Layer Scores

Scoring module:

- `src/catem_score.py`

Layers:

- Physiology: HRV, GSR, ECG
- Workload: NASA-TLX
- System: latency, FPS, tracking
- Performance: task time, errors, success rate

## Phase 4: Dashboard

Technology:

- Streamlit
- Plotly
- Pandas

Pages:

- Overview
- CATEM Score
- Layer Analysis
- Embodiment
- Presence
- Performance
- Physiology
- Workload
- System
- Correlation Heatmaps
- Validation
- Model Comparison

## Phase 5: Validation

Research questions:

- RQ1: Can CATEM integrate multimodal metrics?
- RQ2: Can CATEM predict telepresence quality?
- RQ3: Does CATEM outperform single metrics?
- RQ4: Which CATEM layers matter most?

Evaluation:

- R2
- MAE
- RMSE
- Correlation
- Feature importance

## Phase 6: Publication Assets

Figures:

- CATEM Architecture
- Data Pipeline
- Dataset Schema
- Correlation Matrix
- Feature Importance
- Model Comparison
- Dashboard Screenshot

## Phase 7: Research Paper

Working title:

CATEM: A Cross-Layer Adaptive Telepresence Evaluation Framework Using Multimodal Benchmark Data

