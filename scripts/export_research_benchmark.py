from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "CATEM" / "benchmark" / "catem_v1.csv"
OUTPUT = ROOT / "datasets" / "benchmark" / "catem_master.csv"

MASTER_COLUMNS = [
    "participant_id",
    "session_id",
    "task_id",
    "timestamp",
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
    "telepresence_quality",
]


def main() -> None:
    df = pd.read_csv(SOURCE)
    df["telepresence_quality"] = df["overall_telepresence_quality"]
    master = df.reindex(columns=MASTER_COLUMNS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(master)} rows")


if __name__ == "__main__":
    main()
