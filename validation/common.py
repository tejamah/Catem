from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "datasets" / "benchmark" / "catem_master.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "outputs" / "reports"

MODEL_GROUPS = {
    "presence_only": ["presence_score"],
    "workload_only": ["mental_demand", "effort", "frustration"],
    "system_only": ["latency_ms", "fps", "tracking_error"],
    "performance_only": ["task_completion_time", "error_rate", "success_rate"],
    "catem_full_model": [
        "heart_rate",
        "hrv",
        "gsr",
        "mental_demand",
        "effort",
        "frustration",
        "latency_ms",
        "fps",
        "tracking_error",
        "task_completion_time",
        "error_rate",
        "success_rate",
        "ownership_score",
        "agency_score",
        "presence_score",
    ],
}


def load_master() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def numeric_features(df: pd.DataFrame, exclude_target: bool = True) -> list[str]:
    excluded = {"telepresence_quality"} if exclude_target else set()
    return [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in excluded and not df[col].dropna().empty
    ]


def prepare_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    available = [feature for feature in features if feature in df.columns and not df[feature].dropna().empty]
    X = df[available].fillna(df[available].median(numeric_only=True)).fillna(0)
    y = df["telepresence_quality"].fillna(df["telepresence_quality"].mean())
    return X, y


def write_table(df: pd.DataFrame, filename: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    return path
