import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.benchmark_dataset import BENCHMARK_COLUMNS, catem_to_benchmark_schema, score_benchmark_layers
from src.data_loader import save_data


def main() -> None:
    inputs = [
        ROOT_DIR / "CATEM" / "outputs" / "physionet_catem.csv",
        ROOT_DIR / "CATEM" / "outputs" / "nasa_tlx_catem.csv",
        ROOT_DIR / "CATEM" / "outputs" / "roboturk_benchmark.csv",
        ROOT_DIR / "CATEM" / "outputs" / "ros_catem.csv",
    ]
    frames = []
    for path in inputs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not set(BENCHMARK_COLUMNS).intersection(frame.columns):
            frame = catem_to_benchmark_schema(frame, source_dataset=path.stem)
        frames.append(frame)
    if not frames:
        print("No CATEM ETL outputs found. Run load_* scripts first.")
        return
    merged = pd.concat(frames, ignore_index=True, sort=False)
    benchmark = score_benchmark_layers(merged)
    save_data(benchmark, "CATEM/benchmark/catem_master.csv")
    print(f"Wrote CATEM/benchmark/catem_master.csv with {len(benchmark)} records")


if __name__ == "__main__":
    main()
