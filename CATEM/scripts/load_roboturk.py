import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.benchmark_dataset import load_roboturk_metric_rows, score_benchmark_layers
from src.data_loader import save_data


def main() -> None:
    result_dir = ROOT_DIR / "CATEM" / "datasets" / "roboturk" / "results"
    if not result_dir.exists():
        print(f"Place RoboTurk result CSV files under {result_dir}")
        return
    df = score_benchmark_layers(load_roboturk_metric_rows(result_dir))
    save_data(df, "CATEM/outputs/roboturk_benchmark.csv")
    print("Wrote CATEM/outputs/roboturk_benchmark.csv")


if __name__ == "__main__":
    main()

