from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_path(path: str | Path) -> Path:
    """Resolve project-relative paths consistently across entrypoints."""
    data_path = Path(path)
    if data_path.is_absolute():
        return data_path
    return PROJECT_ROOT / data_path


def load_data(path: str) -> pd.DataFrame:
    """Load CATEM dataset from a CSV file."""
    data_path = resolve_data_path(path)
    sample = pd.read_csv(data_path, nrows=0)
    parse_dates = ["timestamp"] if "timestamp" in sample.columns else []
    df = pd.read_csv(data_path, parse_dates=parse_dates)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file."""
    data_path = resolve_data_path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
