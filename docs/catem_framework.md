# CATEM Framework

CATEM provides a layered evaluation model for telepresence experience.

## Layers

- Embodiment Layer: ownership, agency, self-location
- Presence Layer: presence and social presence
- Behavior Layer: task completion, errors, movement, interaction
- Physiology Layer: heart rate, HRV, GSR, eye fixation, blink rate
- Workload Layer: NASA-TLX, mental demand, effort, frustration
- System Layer: latency, FPS, jitter, packet loss, tracking loss
- Data Quality Layer: missing data, timestamp accuracy, sensor sync

## Prototype Scoring

The prototype now includes two scoring stages:

1. Literature-weighted CATEM score

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

2. Data-driven validation

The dashboard evaluates whether CATEM explains telepresence quality better than single-layer metrics using:

- selected cross-layer correlations
- multiple regression
- random forest feature importance
- actionable score-drop explanations

## Research Correlations

The current validation view highlights:

- Ownership <-> Presence
- Agency <-> Performance
- Latency <-> Agency
- Workload <-> Error Rate

## Explainability

CATEM is designed to be actionable. For low-scoring sessions, the dashboard identifies likely drivers such as high latency, high workload, tracking instability, packet loss, low agency, low presence, or reduced HRV.
