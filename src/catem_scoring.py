import pandas as pd


LITERATURE_WEIGHTS = {
    "embodiment_score": 0.25,
    "presence_score": 0.20,
    "behavior_score": 0.20,
    "physiology_score": 0.10,
    "system_stability_score": 0.15,
    "data_quality_score": 0.10,
    "workload_risk_score": -0.10,
}


def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_value = series.min()
    max_value = series.max()
    if min_value == max_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def compute_layer_score(df: pd.DataFrame, cols: list[str], invert: bool = False) -> pd.Series:
    normalized = pd.DataFrame({col: normalize_series(df[col]) for col in cols if col in df.columns})
    score = normalized.mean(axis=1)
    if invert:
        score = 1 - score
    return score


def compute_catem_scores(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["embodiment_score"] = compute_layer_score(data, ["ownership_score", "agency_score", "self_location_score"])
    data["presence_score"] = compute_layer_score(data, ["presence_score", "social_presence_score"])
    data["behavior_score"] = compute_layer_score(data, ["task_completion_time", "error_rate", "movement_smoothness", "interaction_frequency"], invert=True)
    data["physiology_score"] = compute_layer_score(data, ["heart_rate", "hrv", "gsr", "eye_fixation", "blink_rate"], invert=True)
    data["workload_risk_score"] = compute_layer_score(data, ["nasa_tlx_score", "mental_demand", "effort", "frustration"])
    data["workload_score"] = 1 - data["workload_risk_score"]
    data["system_stability_score"] = compute_layer_score(data, ["latency_ms", "fps", "jitter", "packet_loss", "tracking_loss"], invert=True)
    data["data_quality_score"] = compute_layer_score(data, ["missing_data_rate", "timestamp_accuracy", "sensor_sync_error"], invert=True)

    data["catem_concept_score"] = (
        data["embodiment_score"]
        + data["presence_score"]
        + data["behavior_score"]
        + data["physiology_score"]
        + data["system_stability_score"]
        + data["data_quality_score"]
        - data["workload_score"]
    )
    data["catem_score"] = sum(data[col] * weight for col, weight in LITERATURE_WEIGHTS.items()).clip(0, 1)

    score_cols = [
        "embodiment_score",
        "presence_score",
        "behavior_score",
        "physiology_score",
        "workload_risk_score",
        "workload_score",
        "system_stability_score",
        "data_quality_score",
        "catem_concept_score",
        "catem_score",
    ]
    output_cols = list(dict.fromkeys([*df.columns, *score_cols]))
    return data[output_cols]
