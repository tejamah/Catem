# Data Dictionary

This file documents the synthetic CATEM dataset columns.

- `participant_id`: Unique participant identifier
- `session_id`: Session identifier
- `timestamp`: Timestamp of the measurement
- `ownership_score`: Ownership component of embodiment
- `agency_score`: Agency component of embodiment
- `self_location_score`: Self-location component of embodiment
- `presence_score`: Presence metric
- `social_presence_score`: Social presence metric
- `task_completion_time`: Time to complete task in seconds
- `error_rate`: Task error rate
- `movement_smoothness`: Smoothness of movement
- `interaction_frequency`: Frequency of user interactions
- `heart_rate`: Heart rate in beats per minute
- `hrv`: Heart rate variability
- `gsr`: Galvanic skin response
- `eye_fixation`: Eye fixation duration
- `blink_rate`: Blink frequency
- `nasa_tlx_score`: Overall NASA-TLX score
- `mental_demand`: Mental demand score
- `effort`: Effort score
- `frustration`: Frustration score
- `latency_ms`: System latency in milliseconds
- `fps`: Frames per second
- `jitter`: Network jitter
- `packet_loss`: Packet loss percentage
- `tracking_loss`: Tracking loss percentage
- `missing_data_rate`: Missing data rate
- `timestamp_accuracy`: Timestamp accuracy score
- `sensor_sync_error`: Sensor synchronization error
- `overall_telepresence_quality`: Optional ground-truth or proxy outcome variable for validation. This can come from user satisfaction, immersion, task success, VEQ-style ratings, or dataset-specific proxy labels.

## External Dataset Integration

External sources can be mapped into this schema through `src/dataset_integrations.py`.

- PhysioNet: maps HR, HRV/IBI, EDA/GSR, stress/workload, and missingness into physiology, workload, and data-quality fields.
- OpenNeuro: maps BIDS `*_events.tsv` duration, accuracy, difficulty/load, and interaction counts into behavior and workload fields.
- DEAP: maps valence, arousal, dominance, liking, EEG variance, and peripheral signal variance into presence, workload, agency, physiology, and an optional quality proxy.

External datasets may only cover part of CATEM. Missing layers are filled with neutral defaults until richer telepresence-specific data is available.
