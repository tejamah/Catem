# Paper Outline

Title:

CATEM: A Cross-Layer Adaptive Telepresence Evaluation Framework Using Multimodal Benchmark Data

## 1. Introduction

Motivate the need for holistic telepresence evaluation across human state, system telemetry, task performance, workload, presence, and embodiment.

## 2. Related Work

Discuss telepresence evaluation, embodiment/presence questionnaires, NASA-TLX workload analysis, robot teleoperation datasets, physiological stress datasets, and multimodal human-state analytics.

## 3. CATEM Framework

Define CATEM layers and explain how cross-layer data produces a unified score.

## 4. Dataset Construction

Describe construction of CATEM Benchmark Dataset v1 from NASA-TLX, RoboTurk, CISC-LIVE-LAB-3-style workbook data, robot anomaly telemetry, and future PhysioNet/ROS additions.

## 5. Methodology

Explain ETL, normalization, layer scoring, CATEM score computation, and validation models.

## 6. Validation

Compare presence-only, workload-only, system-only, performance-only, and CATEM full model predictors.

## 7. Results

Report R2, MAE, RMSE, correlation, and feature importance.

## 8. Discussion

Interpret which layers matter most and how CATEM supports decision making.

## 9. Limitations

Current benchmark combines heterogeneous public sources and proxy outcome labels. A controlled telepresence study is needed for stronger causal validation.

## 10. Future Work

Add 1000+ records, integrate real PhysioNet/ROS files, collect direct embodiment/presence questionnaires, and test adaptive teleoperation feedback loops.

