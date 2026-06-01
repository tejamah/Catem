from __future__ import annotations

import pandas as pd

from src.benchmark_dataset import score_benchmark_layers
from src.catem_scoring import compute_catem_scores


def compute_layer_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CATEM layer scores for benchmark-schema data."""
    return score_benchmark_layers(df)


def compute_sample_catem_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CATEM scores for the original sample-data schema."""
    return compute_catem_scores(df)


def physiology_score(df: pd.DataFrame) -> pd.Series:
    return score_benchmark_layers(df)["physiology_score"]


def workload_score(df: pd.DataFrame) -> pd.Series:
    return score_benchmark_layers(df)["workload_score"]


def system_score(df: pd.DataFrame) -> pd.Series:
    return score_benchmark_layers(df)["system_score"]


def performance_score(df: pd.DataFrame) -> pd.Series:
    return score_benchmark_layers(df)["performance_score"]

