import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning and fill missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = df.dropna(subset=["participant_id", "session_id", "timestamp"])
    return df


def normalize_layer(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Scale a list of numeric columns to 0-1 range."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            min_value = df[col].min()
            max_value = df[col].max()
            if min_value != max_value:
                df[col] = (df[col] - min_value) / (max_value - min_value)
            else:
                df[col] = 0.0
    return df
