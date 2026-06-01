import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.benchmark_dataset import load_apple_health_rows, score_benchmark_layers
from src.data_loader import save_data


def main() -> None:
    source = Path.home() / "Downloads" / "apple_health_export" / "export.xml"
    benchmark = score_benchmark_layers(load_apple_health_rows(source))
    save_data(benchmark, "CATEM/outputs/apple_health_benchmark.csv")
    print(f"Wrote CATEM/outputs/apple_health_benchmark.csv with {len(benchmark)} records")


if __name__ == "__main__":
    main()
