# CATEM Benchmark Dataset v1 Card

## Dataset Name

CATEM Benchmark Dataset v1

## Purpose

Provide a unified benchmark schema for developing and testing the CATEM telepresence evaluation pipeline.

## Current Size

4,524 records.

## Sources

- NASA-TLX workload exports
- RoboTurk PSNR/SSIM result curves
- Zenodo multimodal workbook
- Robot anomaly telemetry CSV
- Apple Health wearable/activity XML export

## Schema

The root research file is:

`datasets/benchmark/catem_master.csv`

The target variable is:

`telepresence_quality`

## Collection Methodology

The benchmark is produced by ETL scripts that normalize each source into common identifiers, timestamps, physiology/activity proxies, workload features, system metrics, performance metrics, human-state features, and an outcome proxy.

## Limitations

This is not yet a single controlled telepresence experiment.

The current `telepresence_quality` field is a proxy target assembled from available outcome-like fields. It should not be treated as independent ground truth until validated against real telepresence users.

## Intended Use

- testing CATEM ETL
- comparing model groups
- generating reproducible validation tables
- producing dashboard and publication figure prototypes

## Not Intended Use

- claiming final clinical, HRI, or VR telepresence validity
- claiming superiority over existing telepresence instruments without an independent user-study target
