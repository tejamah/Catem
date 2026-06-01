from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.catem_scoring import compute_catem_scores, normalize_series
from src.data_loader import load_data, save_data
from src.dataset_integrations import combine_catem_sources


BENCHMARK_COLUMNS = [
    "participant_id",
    "session_id",
    "task_id",
    "timestamp",
    "heart_rate",
    "hrv",
    "ecg_signal",
    "gsr",
    "eda",
    "skin_temp",
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "effort",
    "frustration",
    "nasa_tlx_total",
    "latency_ms",
    "tracking_error",
    "packet_loss",
    "jitter",
    "fps",
    "task_completion_time",
    "error_rate",
    "success_rate",
    "path_efficiency",
    "ownership_score",
    "agency_score",
    "presence_score",
    "overall_telepresence_quality",
    "physiology_score",
    "workload_score",
    "system_score",
    "performance_score",
    "catem_score",
]


METRIC_GROUPS = {
    "presence_only": ["presence_score"],
    "workload_only": ["nasa_tlx_total", "mental_demand", "effort", "frustration"],
    "performance_only": ["task_completion_time", "error_rate", "success_rate", "path_efficiency"],
    "system_only": ["latency_ms", "tracking_error", "packet_loss", "jitter", "fps"],
    "physiology_only": ["heart_rate", "hrv", "ecg_signal", "gsr", "eda", "skin_temp"],
    "catem_full_model": ["catem_score"],
}


def catem_to_benchmark_schema(catem_df: pd.DataFrame, source_dataset: str = "catem") -> pd.DataFrame:
    """Map CATEM rows into the CATEM Benchmark Dataset v1 schema."""
    df = catem_df.copy()
    if "catem_score" not in df.columns:
        df = compute_catem_scores(df)

    benchmark = pd.DataFrame()
    benchmark["participant_id"] = df["participant_id"]
    benchmark["session_id"] = df["session_id"]
    benchmark["task_id"] = df.get("task_id", df["session_id"])
    benchmark["timestamp"] = df["timestamp"]
    benchmark["heart_rate"] = df.get("heart_rate")
    benchmark["hrv"] = df.get("hrv")
    benchmark["ecg_signal"] = df.get("ecg_signal", np.nan)
    benchmark["gsr"] = df.get("gsr")
    benchmark["eda"] = df.get("eda", df.get("gsr"))
    benchmark["skin_temp"] = df.get("skin_temp", np.nan)
    benchmark["mental_demand"] = df.get("mental_demand")
    benchmark["physical_demand"] = df.get("physical_demand", np.nan)
    benchmark["temporal_demand"] = df.get("temporal_demand", np.nan)
    benchmark["effort"] = df.get("effort")
    benchmark["frustration"] = df.get("frustration")
    benchmark["nasa_tlx_total"] = df.get("nasa_tlx_total", df.get("nasa_tlx_score"))
    benchmark["latency_ms"] = df.get("latency_ms")
    benchmark["tracking_error"] = df.get("tracking_error", df.get("tracking_loss"))
    benchmark["packet_loss"] = df.get("packet_loss")
    benchmark["jitter"] = df.get("jitter")
    benchmark["fps"] = df.get("fps")
    benchmark["task_completion_time"] = df.get("task_completion_time")
    benchmark["error_rate"] = df.get("error_rate")
    benchmark["success_rate"] = df.get("success_rate", 1 - df.get("error_rate", np.nan))
    benchmark["path_efficiency"] = df.get("path_efficiency", df.get("movement_smoothness"))
    benchmark["ownership_score"] = df.get("ownership_score")
    benchmark["agency_score"] = df.get("agency_score")
    benchmark["presence_score"] = df.get("presence_score")
    benchmark["overall_telepresence_quality"] = df.get("overall_telepresence_quality")
    benchmark["source_dataset"] = source_dataset
    return benchmark


def score_benchmark_layers(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Create benchmark layer scores and final CATEM score."""
    df = benchmark_df.copy()
    df["physiology_score"] = pd.DataFrame(
        {
            "hrv": normalize_series(df["hrv"]),
            "gsr": 1 - normalize_series(df["gsr"]),
            "heart_rate": 1 - normalize_series(df["heart_rate"]),
            "eda": 1 - normalize_series(df["eda"]),
        }
    ).mean(axis=1).fillna(0.5)
    df["workload_score"] = 1 - pd.DataFrame(
        {
            "nasa_tlx_total": normalize_series(df["nasa_tlx_total"]),
            "mental_demand": normalize_series(df["mental_demand"]),
            "effort": normalize_series(df["effort"]),
            "frustration": normalize_series(df["frustration"]),
        }
    ).mean(axis=1).fillna(0.5)
    df["system_score"] = pd.DataFrame(
        {
            "latency": 1 - normalize_series(df["latency_ms"]),
            "tracking": 1 - normalize_series(df["tracking_error"]),
            "packet_loss": 1 - normalize_series(df["packet_loss"]),
            "jitter": 1 - normalize_series(df["jitter"]),
            "fps": normalize_series(df["fps"]),
        }
    ).mean(axis=1).fillna(0.5)
    df["performance_score"] = pd.DataFrame(
        {
            "task_time": 1 - normalize_series(df["task_completion_time"]),
            "error_rate": 1 - normalize_series(df["error_rate"]),
            "success_rate": normalize_series(df["success_rate"]),
            "path_efficiency": normalize_series(df["path_efficiency"]),
        }
    ).mean(axis=1).fillna(0.5)
    df["catem_score"] = (
        df[["ownership_score", "agency_score", "presence_score"]].mean(axis=1).fillna(0.5)
        + df["performance_score"]
        + df["physiology_score"]
        + df["system_score"]
        - (1 - df["workload_score"])
    ).clip(0, 4) / 4
    if "overall_telepresence_quality" not in df or df["overall_telepresence_quality"].isna().all():
        df["overall_telepresence_quality"] = df[
            ["presence_score", "agency_score", "success_rate", "performance_score", "catem_score"]
        ].mean(axis=1)
    return df[BENCHMARK_COLUMNS]


def normalize_to_long_format(benchmark_df: pd.DataFrame, source_dataset: str = "catem_benchmark_v1") -> pd.DataFrame:
    """Convert the wide benchmark table to traceable long metric format."""
    id_cols = ["participant_id", "session_id", "task_id", "timestamp"]
    metric_cols = [col for col in benchmark_df.columns if col not in id_cols]
    long_df = benchmark_df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="metric_value",
    )
    long_df["source_dataset"] = source_dataset
    return long_df.dropna(subset=["metric_value"])


def simple_r2(features: pd.DataFrame, target: pd.Series) -> float:
    if features.empty or len(features) < 2:
        return float("nan")
    X = features.fillna(features.median(numeric_only=True)).fillna(0)
    y = target.fillna(target.mean())
    model = LinearRegression()
    model.fit(X, y)
    return float(model.score(X, y))


def validate_benchmark(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Compare CATEM against single-layer metric groups."""
    target = benchmark_df["overall_telepresence_quality"]
    rows = []
    for model_name, features in METRIC_GROUPS.items():
        available = [feature for feature in features if feature in benchmark_df.columns]
        if not available:
            continue
        group_score = benchmark_df[available].mean(axis=1)
        rows.append(
            {
                "model": model_name,
                "pearson_correlation": float(group_score.corr(target)),
                "r2": simple_r2(benchmark_df[available], target),
                "feature_count": len(available),
            }
        )

    candidate_features = [
        "heart_rate",
        "hrv",
        "gsr",
        "nasa_tlx_total",
        "latency_ms",
        "tracking_error",
        "packet_loss",
        "fps",
        "task_completion_time",
        "error_rate",
        "success_rate",
        "agency_score",
        "presence_score",
    ]
    available = [feature for feature in candidate_features if feature in benchmark_df.columns]
    if len(available) >= 2 and len(benchmark_df) >= 2:
        X = benchmark_df[available].fillna(benchmark_df[available].median(numeric_only=True)).fillna(0)
        y = target.fillna(target.mean())
        forest = RandomForestRegressor(random_state=42, n_estimators=250)
        forest.fit(X, y)
        for feature, importance in sorted(zip(available, forest.feature_importances_), key=lambda item: item[1], reverse=True):
            rows.append(
                {
                    "model": f"feature_importance::{feature}",
                    "pearson_correlation": np.nan,
                    "r2": float(importance),
                    "feature_count": 1,
                }
            )
    return pd.DataFrame(rows)


def build_catem_benchmark_v1(
    source_frames: list[pd.DataFrame] | None = None,
    output_path: str | Path = "data/processed/catem_benchmark_v1.csv",
    long_output_path: str | Path = "data/processed/catem_benchmark_v1_long.csv",
    validation_output_path: str | Path = "outputs/catem_benchmark_validation.csv",
) -> pd.DataFrame:
    """Build CATEM Benchmark Dataset v1 from source frames or project sample data."""
    if source_frames:
        combined = combine_catem_sources(*source_frames)
    else:
        combined = load_data("data/synthetic/catem_sample_data.csv")

    benchmark = catem_to_benchmark_schema(combined, source_dataset="catem_benchmark_v1")
    benchmark = score_benchmark_layers(benchmark)
    long_df = normalize_to_long_format(benchmark)
    validation_df = validate_benchmark(benchmark)

    save_data(benchmark, output_path)
    save_data(long_df, long_output_path)
    save_data(validation_df, validation_output_path)
    return benchmark
