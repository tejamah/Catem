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

The prototype combines layer scores into a single CATEM score.
Workload is currently treated as a subtraction factor in the prototype model.
