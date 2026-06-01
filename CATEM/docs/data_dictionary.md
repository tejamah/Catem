# CATEM Data Dictionary

| Feature | Layer | Description |
| --- | --- | --- |
| `participant_id` | Identifier | Participant, operator, robot, or subject identifier |
| `session_id` | Identifier | Session, scenario, environment, or condition |
| `task_id` | Identifier | Task name or benchmark task group |
| `timestamp` | Identifier | Time of observation or aggregation |
| `heart_rate` | Physiology | Heart activity or pulse rate |
| `hrv` | Physiology | Heart rate variability |
| `ecg_signal` | Physiology | ECG-derived signal or feature |
| `gsr` | Physiology | Galvanic skin response |
| `eda` | Physiology | Electrodermal activity |
| `skin_temp` | Physiology | Skin or wearable temperature |
| Apple Health activity proxy | Physiology | Daily proxy derived from steps, active energy, walking speed, walking steadiness, asymmetry, double-support percentage, and sleep records |
| `mental_demand` | Workload | NASA-TLX mental demand |
| `physical_demand` | Workload | NASA-TLX physical demand |
| `temporal_demand` | Workload | NASA-TLX temporal demand |
| `effort` | Workload | NASA-TLX effort |
| `frustration` | Workload | NASA-TLX frustration |
| `nasa_tlx_total` | Workload | Overall workload score |
| `latency_ms` | System | Network or control delay |
| `tracking_error` | System | Tracking, scan, or state-estimation error |
| `packet_loss` | System | Packet loss or low-message-rate proxy |
| `jitter` | System | Timing or latency variability |
| `fps` | System | Frame rate, message rate, or control stream frequency |
| `task_completion_time` | Performance | Time to complete a task |
| `error_rate` | Performance | Task, anomaly, or prediction error rate |
| `success_rate` | Performance | Success or non-anomaly rate |
| `path_efficiency` | Performance | Motion/path efficiency or visual quality proxy |
| `ownership_score` | Embodiment | Sense of body ownership |
| `agency_score` | Embodiment | Sense of control or agency |
| `presence_score` | Presence | Presence, situation awareness, or immersion proxy |
| `overall_telepresence_quality` | Outcome | Ground-truth or proxy target for validation |
| `physiology_score` | CATEM Layer Score | Normalized physiology layer |
| `workload_score` | CATEM Layer Score | Inverted workload layer where higher is better |
| `system_score` | CATEM Layer Score | Normalized system stability layer |
| `performance_score` | CATEM Layer Score | Normalized task performance layer |
| `catem_score` | CATEM Score | Unified CATEM evaluation score |
