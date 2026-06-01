# CATEM Analytics Platform Roadmap

CATEM should evolve from an evaluation framework into a decision-support platform for telepresence research and operation.

## Primary Product Direction

### Option 1: Telepresence Evaluation Platform

This is the most natural path from the current project.

Target users:

- HRI researchers
- VR researchers
- robotics labs
- university research groups

Problem:

Researchers need a holistic way to evaluate telepresence systems across embodiment, presence, workload, task performance, physiology, system telemetry, and data quality.

Input:

- VEQ or embodiment questionnaires
- presence questionnaires
- NASA-TLX workload surveys
- telemetry logs
- task-performance logs
- optional physiological data

Output:

- telepresence quality score
- layer scores
- root-cause analysis
- improvement recommendations
- validation reports

## Secondary Applications

### Real-Time Teleoperation Monitoring

CATEM can monitor latency, errors, workload, physiology, and tracking instability to predict performance degradation before a task failure occurs.

### XR Training Analytics

CATEM can support medical, military, and industrial XR training by estimating whether trainees are immersed, embodied, overloaded, or performing effectively.

## Technical Roadmap

### Phase 1: Real Data

Replace or supplement synthetic data with Cornell telepresence data, public telepresence datasets, or controlled experimental data.

### Phase 2: Data Pipeline

```text
Data Collection
-> ETL Pipeline
-> CATEM Engine
-> Database
-> Dashboard
-> Decision Support
```

Suggested stack:

- Python
- FastAPI
- PostgreSQL
- Streamlit
- Plotly
- Docker

### Phase 3: Explainable AI

CATEM should not only output a score. It should explain why the score changed.

Example:

```text
CATEM Score = 68

Drivers:
- agency decreased
- presence decreased
- workload increased
- latency increased
```

### Phase 4: Recommendations

CATEM should translate detected risks into system-level actions.

Examples:

- high latency -> reduce network load or optimize streaming
- high workload -> simplify interface or reduce concurrent tasks
- tracking instability -> recalibrate sensors
- low agency -> improve control responsiveness
- low presence -> improve spatial, visual, or social cues

## Publication Path

Paper 1:

CATEM: A Cross-Layer Framework for Telepresence Evaluation

Paper 2:

Validation of CATEM Using Multimodal Telepresence Data

Platform:

CATEM Analytics Platform for Telepresence Decision Support
