# CATEM Telepresence Evaluation

CATEM is a Cross-Layer Adaptive Telepresence Evaluation Model for modeling, validating, and visualizing telepresence quality across human, behavioral, physiological, system, and data-quality factors.

The project is moving from a framework and dashboard toward a telepresence decision-support platform: researchers upload multimodal study data, CATEM scores the session, explains quality risks, and recommends improvements.

## What CATEM Integrates

- Embodiment: ownership, agency, self-location
- Presence: spatial presence and social presence
- Behavior: task completion, errors, movement, interaction
- Physiology: heart rate, HRV, GSR, eye fixation, blink rate
- Workload: NASA-TLX, mental demand, effort, frustration
- System: latency, FPS, jitter, packet loss, tracking loss
- Data Quality: missing data, timestamp accuracy, sensor synchronization

## Scoring Model

The current implementation uses a literature-weighted score:

```text
CATEM =
0.25 Embodiment
+ 0.20 Presence
+ 0.20 Behavior
+ 0.10 Physiology
+ 0.15 System
+ 0.10 Data Quality
- 0.10 Workload Risk
```

The dashboard also includes data-driven validation tools: correlation analysis, feature importance, explainability for low-scoring sessions, and recommendation generation.

## External Dataset Integration

CATEM includes local adapters for public physiology and neuroscience datasets:

- PhysioNet: ECG, heart rate, HRV/IBI, EDA/GSR, stress, and wearable sensor exports
- ROS/robot logs: latency, FPS/rate, jitter, packet loss, tracking loss, motion, and task telemetry
- NASA-TLX: workload, mental demand, effort, and frustration survey exports
- OpenNeuro: BIDS EEG/behavioral experiment folders with `*_events.tsv` files
- DEAP: preprocessed `.dat` files with EEG, peripheral physiology, and affective labels

Place downloaded datasets under `data/external/` and use `src.dataset_integrations` to map them into the CATEM schema. See `docs/external_dataset_integration.md` for examples.

The recommended research approach is not to search for one perfect dataset. Instead, combine complementary sources into a harmonized CATEM dataset:

```text
PhysioNet physiology + ROS telemetry + NASA-TLX workload + outcome labels
-> CATEM Dataset
-> scoring, validation, explainability, and recommendations
```

## Decision Support

CATEM is designed to convert analytics into action:

- high latency -> reduce network load or optimize streaming
- high workload -> simplify the interface or reduce concurrent task demands
- tracking instability -> recalibrate sensors
- low agency -> improve control responsiveness and feedback timing
- low presence -> strengthen spatial, visual, audio, or social cues

## Project Structure

- `app.py` - prototype runner for data loading and scoring
- `data/` - raw, processed, and synthetic datasets
- `src/` - loading, preprocessing, scoring, validation, and visualization modules
- `dashboards/` - Streamlit dashboard implementation
- `notebooks/` - exploratory and validation notebooks
- `outputs/` - generated figures, reports, and results
- `docs/` - CATEM framework and validation documentation
  - `docs/platform_roadmap.md` - path from framework to decision-support platform
  - `docs/external_dataset_integration.md` - PhysioNet, OpenNeuro, and DEAP integration guide
  - `docs/catem_dataset_strategy.md` - strategy for combining partial datasets into one CATEM dataset

## Getting Started

1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the prototype app

```bash
python app.py
```

3. Start the dashboard

```bash
streamlit run dashboards/streamlit_dashboard.py
```

## Research Direction

This repository represents the CATEM implementation stage. The next validation step is to map real Cornell telepresence data into the same layers, compare CATEM against single-layer metrics, and report learned model weights from regression or tree-based models.

The intended project evolution is:

```text
Framework
-> Decision Support System
-> Real Users
-> Real Value
```
