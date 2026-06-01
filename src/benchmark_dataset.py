from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
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


def ensure_benchmark_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a benchmark-shaped frame, filling unavailable modalities with NaN."""
    benchmark = df.copy()
    for col in BENCHMARK_COLUMNS:
        if col not in benchmark.columns:
            benchmark[col] = np.nan
    return benchmark[BENCHMARK_COLUMNS]


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
    df = ensure_benchmark_columns(benchmark_df)
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
    return ensure_benchmark_columns(df)


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
        feature_frame = benchmark_df[available]
        if feature_frame.dropna(how="all").empty:
            continue
        group_score = benchmark_df[available].mean(axis=1)
        if group_score.dropna().shape[0] < 2:
            continue
        valid = group_score.notna() & target.notna()
        prediction = group_score[valid]
        actual = target[valid]
        mae = float((prediction - actual).abs().mean())
        rmse = float(np.sqrt(((prediction - actual) ** 2).mean()))
        rows.append(
            {
                "model": model_name,
                "pearson_correlation": float(group_score.corr(target)),
                "r2": simple_r2(benchmark_df[available], target),
                "mae": mae,
                "rmse": rmse,
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
    available = [
        feature
        for feature in candidate_features
        if feature in benchmark_df.columns and not benchmark_df[feature].dropna().empty
    ]
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
                    "mae": np.nan,
                    "rmse": np.nan,
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


def normalize_series_with_bounds(series: pd.Series, low: float | None = None, high: float | None = None) -> pd.Series:
    min_value = series.min() if low is None else low
    max_value = series.max() if high is None else high
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=series.index)
    return ((series - min_value) / (max_value - min_value)).clip(0, 1)


def load_tlx_benchmark_rows(path: str | Path, condition: str) -> pd.DataFrame:
    """Map NASA-TLX clinician CSV exports into CATEM Benchmark v1 rows."""
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        mental = float(row.get("mental_demand", np.nan))
        physical = float(row.get("physical_demand", np.nan))
        temporal = float(row.get("temporal_demand", np.nan))
        effort = float(row.get("effort", np.nan))
        frustration = float(row.get("frustration", np.nan))
        performance_raw = float(row.get("performance", np.nan))
        workload_values = [mental, physical, temporal, effort, frustration]
        tlx_total = float(np.nanmean(workload_values))
        performance_score = normalize_series_with_bounds(pd.Series([performance_raw]), 1, 20).iloc[0]
        rows.append(
            {
                "participant_id": f"clinician_{int(row['physician_number']):02d}",
                "session_id": condition,
                "task_id": f"nasa_tlx_{condition}",
                "timestamp": pd.Timestamp("2026-01-01"),
                "mental_demand": mental,
                "physical_demand": physical,
                "temporal_demand": temporal,
                "effort": effort,
                "frustration": frustration,
                "nasa_tlx_total": tlx_total,
                "success_rate": performance_score,
                "overall_telepresence_quality": performance_score,
            }
        )
    return pd.DataFrame(rows)


def load_roboturk_metric_rows(result_dir: str | Path) -> pd.DataFrame:
    """Map RoboTurk PSNR/SSIM result curves into benchmark rows."""
    result_path = Path(result_dir)
    task_files = {
        "bair_action": ("psnr_bair_action.csv", "ssim_bair_action.csv"),
        "sawyer_laundry_layout": ("psnr_laundry_layout.csv", "ssim_laundry_layout.csv"),
        "sawyer_tower_creation": ("psnr_tower_creation.csv", "ssim_tower_creation.csv"),
    }
    rows = []
    for task_id, (psnr_file, ssim_file) in task_files.items():
        psnr = pd.read_csv(result_path / psnr_file, header=None)[0].astype(float)
        ssim = pd.read_csv(result_path / ssim_file, header=None)[0].astype(float)
        psnr_quality = normalize_series_with_bounds(psnr, 15, 35)
        ssim_quality = normalize_series_with_bounds(ssim, 0.5, 1.0)
        for frame_index, (psnr_value, ssim_value) in enumerate(zip(psnr, ssim)):
            quality = float(np.mean([psnr_quality.iloc[frame_index], ssim_quality.iloc[frame_index]]))
            rows.append(
                {
                    "participant_id": "roboturk_operator_pool",
                    "session_id": task_id,
                    "task_id": task_id,
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=int(frame_index)),
                    "tracking_error": float(1 - ssim_value),
                    "error_rate": float(1 - ssim_quality.iloc[frame_index]),
                    "success_rate": float(ssim_quality.iloc[frame_index]),
                    "path_efficiency": float(psnr_quality.iloc[frame_index]),
                    "overall_telepresence_quality": quality,
                }
            )
    return pd.DataFrame(rows)


def load_robot_anomaly_rows(path: str | Path) -> pd.DataFrame:
    """Map robot anomaly telemetry into CATEM Benchmark v1 rows."""
    raw = pd.read_csv(path)
    rows = []
    gyro_abs = raw["imu_gyro_z"].abs()
    latency_series = pd.to_numeric(raw.get("network_latency_ms", pd.Series(np.nan, index=raw.index)), errors="coerce")
    message_rate_series = pd.to_numeric(raw.get("message_rate_hz", pd.Series(np.nan, index=raw.index)), errors="coerce")
    grouped_latency_std = raw.groupby(["robot_namespace", "environment"], dropna=False)["network_latency_ms"].transform("std")
    for _, row in raw.iterrows():
        anomaly_rate = float(row.get("anomaly", 0))
        scan_quality = float(row.get("scan_valid_pct", np.nan))
        message_rate = float(row.get("message_rate_hz", np.nan))
        latency = float(row.get("network_latency_ms", np.nan))
        cpu_usage = float(row.get("cpu_usage", np.nan))
        battery_pct = float(row.get("battery_pct", np.nan))
        motion_stability = 1 - normalize_series_with_bounds(
            pd.Series([abs(float(row.get("imu_gyro_z", np.nan)))]),
            float(gyro_abs.min()),
            float(gyro_abs.max()),
        ).iloc[0]
        latency_quality = 1 - normalize_series_with_bounds(
            pd.Series([latency]),
            float(latency_series.min()),
            float(latency_series.max()),
        ).iloc[0]
        message_quality = normalize_series_with_bounds(
            pd.Series([message_rate]),
            float(message_rate_series.min()),
            float(message_rate_series.max()),
        ).iloc[0]
        success_rate = float(np.clip(1 - anomaly_rate, 0, 1))
        rows.append(
            {
                "participant_id": str(row.get("robot_namespace", "robot")),
                "session_id": str(row.get("environment", "environment")),
                "task_id": f"robot_anomaly_{row.get('environment', 'environment')}",
                "timestamp": pd.to_datetime(row.get("timestamp"), errors="coerce"),
                "latency_ms": latency,
                "tracking_error": float(1 - scan_quality),
                "packet_loss": float(1 - message_quality),
                "jitter": float(grouped_latency_std.loc[row.name]) if pd.notna(grouped_latency_std.loc[row.name]) else np.nan,
                "fps": message_rate,
                "error_rate": anomaly_rate,
                "success_rate": success_rate,
                "path_efficiency": motion_stability,
                "overall_telepresence_quality": float(
                    np.nanmean([success_rate, scan_quality, motion_stability, battery_pct / 100, latency_quality])
                ),
                "cpu_usage": cpu_usage,
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_from_user_files(
    demographics_path: str | Path,
    tlx_assistant_path: str | Path,
    tlx_current_path: str | Path,
    roboturk_results_dir: str | Path,
    zenodo_workbook_path: str | Path | None = None,
    robot_anomaly_path: str | Path | None = None,
    output_path: str | Path = "data/processed/catem_benchmark_user_data.csv",
    long_output_path: str | Path = "data/processed/catem_benchmark_user_data_long.csv",
    validation_output_path: str | Path = "outputs/catem_benchmark_user_validation.csv",
) -> pd.DataFrame:
    """Build a CATEM benchmark from the provided TLX and RoboTurk files."""
    _demographics = pd.read_csv(demographics_path)
    tlx_assistant = load_tlx_benchmark_rows(tlx_assistant_path, "assistant")
    tlx_current = load_tlx_benchmark_rows(tlx_current_path, "current")
    roboturk = load_roboturk_metric_rows(roboturk_results_dir)
    source_frames = [tlx_assistant, tlx_current, roboturk]
    if zenodo_workbook_path is not None and Path(zenodo_workbook_path).exists():
        source_frames.append(load_zenodo_human_state_rows(zenodo_workbook_path))
    if robot_anomaly_path is not None and Path(robot_anomaly_path).exists():
        source_frames.append(load_robot_anomaly_rows(robot_anomaly_path))

    raw_benchmark = pd.concat(source_frames, ignore_index=True)
    benchmark = score_benchmark_layers(raw_benchmark)
    long_df = normalize_to_long_format(benchmark, source_dataset="catem_user_benchmark")
    validation_df = validate_benchmark(benchmark)

    save_data(benchmark, output_path)
    save_data(long_df, long_output_path)
    save_data(validation_df, validation_output_path)
    return benchmark


def excel_column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return index - 1


def read_xlsx_sheet(path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Read a simple xlsx sheet using stdlib XML parsing."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", ns)))

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        sheet_names = [sheet.attrib["name"] for sheet in workbook_root.findall("a:sheets/a:sheet", ns)]
        if sheet_name not in sheet_names:
            raise ValueError(f"Sheet {sheet_name!r} not found. Available sheets: {sheet_names}")
        sheet_index = sheet_names.index(sheet_name) + 1
        sheet_root = ET.fromstring(workbook.read(f"xl/worksheets/sheet{sheet_index}.xml"))

        parsed_rows = []
        for row in sheet_root.findall("a:sheetData/a:row", ns):
            values = {}
            for cell in row.findall("a:c", ns):
                cell_ref = cell.attrib.get("r", "")
                value_node = cell.find("a:v", ns)
                value = "" if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value != "":
                    value = shared_strings[int(value)]
                values[excel_column_index(cell_ref)] = value
            if values:
                width = max(values) + 1
                parsed_rows.append([values.get(idx, "") for idx in range(width)])

    if not parsed_rows:
        return pd.DataFrame()
    header = parsed_rows[0]
    rows = [row + [""] * (len(header) - len(row)) for row in parsed_rows[1:]]
    df = pd.DataFrame(rows, columns=header).replace("", np.nan)
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            df[col] = numeric
    return df


def load_zenodo_human_state_rows(path: str | Path) -> pd.DataFrame:
    """Map the provided Zenodo multimodal workbook into CATEM Benchmark v1 rows."""
    participant_info = read_xlsx_sheet(path, "participant_info")
    tlx = read_xlsx_sheet(path, "nasa_tlx")
    sart = read_xlsx_sheet(path, "sart")
    spam = read_xlsx_sheet(path, "spam")
    simulator = read_xlsx_sheet(path, "simulator_logs")
    watch = read_xlsx_sheet(path, "watch_data")
    ai_questions = read_xlsx_sheet(path, "AI_questions")

    keys = ["Participant", "Group", "Scenario"]
    merged = tlx.merge(sart, on=keys, how="outer", suffixes=("", "_sart"))
    merged = merged.merge(spam, on=keys, how="outer", suffixes=("", "_spam"))
    merged = merged.merge(simulator, on=keys, how="outer", suffixes=("", "_sim"))
    merged = merged.merge(watch, on=keys, how="outer", suffixes=("", "_watch"))
    merged = merged.merge(ai_questions, on=keys, how="outer", suffixes=("", "_ai"))
    merged = merged.merge(participant_info[["Participant", "Group"]].drop_duplicates(), on=["Participant", "Group"], how="left")

    def norm(value: float, series: pd.Series) -> float:
        numeric = pd.to_numeric(series, errors="coerce")
        if pd.isna(value) or numeric.dropna().empty:
            return np.nan
        return normalize_series_with_bounds(pd.Series([float(value)]), float(numeric.min()), float(numeric.max())).iloc[0]

    def safe_mean(values: list[float]) -> float:
        valid = [value for value in values if pd.notna(value)]
        return float(np.mean(valid)) if valid else np.nan

    rows = []
    for _, row in merged.iterrows():
        participant = str(row.get("Participant", "zenodo"))
        scenario = str(row.get("Scenario", "scenario"))
        accuracy = row.get("Accuracy", np.nan)
        overall_performance = row.get("Overall Performance", np.nan)
        sart_index = row.get("SART_index", np.nan)
        ai_trust = row.get("AI_trust", np.nan)
        performance_proxy = safe_mean(
            [
                norm(accuracy, merged["Accuracy"]),
                norm(overall_performance, merged["Overall Performance"]),
            ]
        )
        presence_proxy = norm(sart_index, merged["SART_index"]) if pd.notna(sart_index) else np.nan
        agency_proxy = norm(ai_trust, merged["AI_trust"]) if pd.notna(ai_trust) else presence_proxy
        rows.append(
            {
                "participant_id": participant,
                "session_id": scenario,
                "task_id": f"zenodo_{scenario}",
                "timestamp": pd.Timestamp("2026-01-01"),
                "heart_rate": row.get("Pulse_rate", np.nan),
                "gsr": row.get("EDA", np.nan),
                "eda": row.get("EDA", np.nan),
                "skin_temp": row.get("Temperature", np.nan),
                "mental_demand": row.get("Mental_demand", np.nan),
                "physical_demand": row.get("Physical_demand", np.nan),
                "temporal_demand": row.get("Temporal_demand", np.nan),
                "effort": row.get("Effort", np.nan),
                "frustration": row.get("Frustration", np.nan),
                "nasa_tlx_total": row.get("TLX_index", np.nan),
                "task_completion_time": row.get("Response_time", row.get("Reaction_time", np.nan)),
                "error_rate": 1 - norm(accuracy, merged["Accuracy"]) if pd.notna(accuracy) else np.nan,
                "success_rate": norm(accuracy, merged["Accuracy"]) if pd.notna(accuracy) else np.nan,
                "path_efficiency": norm(overall_performance, merged["Overall Performance"]) if pd.notna(overall_performance) else np.nan,
                "agency_score": agency_proxy,
                "presence_score": presence_proxy,
                "overall_telepresence_quality": safe_mean([performance_proxy, presence_proxy, agency_proxy]),
            }
        )
    return pd.DataFrame(rows)
