from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import resolve_data_path, save_data


CATEM_COLUMNS = [
    "participant_id",
    "session_id",
    "timestamp",
    "ownership_score",
    "agency_score",
    "self_location_score",
    "presence_score",
    "social_presence_score",
    "task_completion_time",
    "error_rate",
    "movement_smoothness",
    "interaction_frequency",
    "heart_rate",
    "hrv",
    "gsr",
    "eye_fixation",
    "blink_rate",
    "nasa_tlx_score",
    "mental_demand",
    "effort",
    "frustration",
    "latency_ms",
    "fps",
    "jitter",
    "packet_loss",
    "tracking_loss",
    "missing_data_rate",
    "timestamp_accuracy",
    "sensor_sync_error",
    "overall_telepresence_quality",
]


ALIASES = {
    "heart_rate": ["heart_rate", "hr", "HR", "pulse", "bpm"],
    "hrv": ["hrv", "HRV", "ibi", "IBI", "rr_interval", "rr"],
    "gsr": ["gsr", "GSR", "eda", "EDA", "electrodermal_activity"],
    "task_completion_time": ["task_completion_time", "duration", "response_time", "reaction_time", "rt"],
    "error_rate": ["error_rate", "errors", "incorrect", "error"],
    "nasa_tlx_score": ["nasa_tlx_score", "tlx", "workload", "stress", "arousal"],
    "presence_score": ["presence_score", "presence", "immersion", "valence"],
    "agency_score": ["agency_score", "agency", "control_rating"],
    "latency_ms": ["latency_ms", "latency", "delay_ms"],
    "packet_loss": ["packet_loss", "loss"],
    "tracking_loss": ["tracking_loss", "tracking_error"],
}


DEFAULTS = {
    "ownership_score": 0.5,
    "agency_score": 0.5,
    "self_location_score": 0.5,
    "presence_score": 0.5,
    "social_presence_score": 0.5,
    "task_completion_time": 0.0,
    "error_rate": 0.0,
    "movement_smoothness": 0.5,
    "interaction_frequency": 0.0,
    "heart_rate": 0.0,
    "hrv": 0.0,
    "gsr": 0.0,
    "eye_fixation": 0.0,
    "blink_rate": 0.0,
    "nasa_tlx_score": 0.0,
    "mental_demand": 0.0,
    "effort": 0.0,
    "frustration": 0.0,
    "latency_ms": 0.0,
    "fps": 0.0,
    "jitter": 0.0,
    "packet_loss": 0.0,
    "tracking_loss": 0.0,
    "missing_data_rate": 0.0,
    "timestamp_accuracy": 1.0,
    "sensor_sync_error": 0.0,
}


def first_matching_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {col.lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def normalize_to_unit(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1))


def build_base_row(participant_id: str, session_id: str, timestamp: str | pd.Timestamp | None = None) -> dict:
    row = dict(DEFAULTS)
    row["participant_id"] = participant_id
    row["session_id"] = session_id
    row["timestamp"] = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.now()
    row["overall_telepresence_quality"] = np.nan
    return row


def finalize_catem_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in CATEM_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS.get(col, np.nan)
    return df[CATEM_COLUMNS]


def integrate_physionet_wearable_csv(path: str | Path, participant_id: str = "physionet") -> pd.DataFrame:
    """Map a local PhysioNet wearable/stress CSV export into CATEM rows."""
    data_path = resolve_data_path(path)
    raw = pd.read_csv(data_path)
    row = build_base_row(participant_id, data_path.stem)
    for target, aliases in ALIASES.items():
        col = first_matching_column(raw, aliases)
        if col is not None and pd.api.types.is_numeric_dtype(raw[col]):
            row[target] = float(raw[col].dropna().mean())

    if "heart_rate" in row and row["heart_rate"]:
        row["physiology_source"] = "PhysioNet"
    if "hrv" in row and row["hrv"]:
        row["hrv"] = float(raw[first_matching_column(raw, ALIASES["hrv"])].dropna().std())
    row["missing_data_rate"] = float(raw.isna().mean().mean())
    return finalize_catem_frame([row])


def integrate_openneuro_bids(root: str | Path) -> pd.DataFrame:
    """Map local OpenNeuro/BIDS events TSV files into CATEM rows."""
    root_path = resolve_data_path(root)
    rows = []
    for events_path in root_path.rglob("*_events.tsv"):
        events = pd.read_csv(events_path, sep="\t")
        participant = next((part for part in events_path.parts if part.startswith("sub-")), "openneuro")
        session = next((part for part in events_path.parts if part.startswith("ses-")), events_path.stem)
        row = build_base_row(participant, session)

        duration_col = first_matching_column(events, ["duration", "response_time", "reaction_time", "rt"])
        if duration_col and pd.api.types.is_numeric_dtype(events[duration_col]):
            row["task_completion_time"] = float(events[duration_col].dropna().mean())

        accuracy_col = first_matching_column(events, ["accuracy", "correct", "response_accuracy"])
        if accuracy_col and pd.api.types.is_numeric_dtype(events[accuracy_col]):
            row["error_rate"] = float(1 - events[accuracy_col].dropna().mean())

        workload_col = first_matching_column(events, ["difficulty", "load", "workload", "condition"])
        if workload_col and pd.api.types.is_numeric_dtype(events[workload_col]):
            row["nasa_tlx_score"] = float(events[workload_col].dropna().mean())

        row["interaction_frequency"] = float(len(events))
        row["missing_data_rate"] = float(events.isna().mean().mean())
        rows.append(row)
    return finalize_catem_frame(rows)


def integrate_deap_preprocessed_dat(path: str | Path, participant_id: str | None = None) -> pd.DataFrame:
    """Map one DEAP preprocessed .dat participant file into CATEM trial rows."""
    data_path = resolve_data_path(path)
    with data_path.open("rb") as file:
        payload = pickle.load(file, encoding="latin1")

    signals = np.asarray(payload["data"])
    labels = np.asarray(payload["labels"])
    participant = participant_id or data_path.stem
    rows = []
    for trial_index, trial in enumerate(signals):
        row = build_base_row(participant, f"trial_{trial_index + 1:02d}")
        valence, arousal, dominance, liking = labels[trial_index, :4]
        eeg = trial[:32]
        peripheral = trial[32:]

        row["presence_score"] = normalize_to_unit(valence, 1, 9)
        row["social_presence_score"] = normalize_to_unit(liking, 1, 9)
        row["agency_score"] = normalize_to_unit(dominance, 1, 9)
        row["nasa_tlx_score"] = normalize_to_unit(arousal, 1, 9)
        row["mental_demand"] = row["nasa_tlx_score"]
        row["effort"] = row["nasa_tlx_score"]
        row["frustration"] = row["nasa_tlx_score"]
        row["gsr"] = float(np.nanstd(peripheral)) if peripheral.size else 0.0
        row["hrv"] = float(np.nanstd(eeg))
        row["heart_rate"] = float(np.nanmean(np.abs(peripheral))) if peripheral.size else 0.0
        row["overall_telepresence_quality"] = float(np.mean([row["presence_score"], row["agency_score"], row["social_presence_score"]]))
        rows.append(row)
    return finalize_catem_frame(rows)


def save_integrated_dataset(df: pd.DataFrame, output_path: str | Path = "data/processed/catem_external_integrated.csv") -> None:
    save_data(df, output_path)
