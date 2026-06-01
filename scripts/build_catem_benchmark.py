import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.benchmark_dataset import build_catem_benchmark_v1


def main() -> None:
    benchmark = build_catem_benchmark_v1()
    print(f"Built CATEM Benchmark Dataset v1 with {len(benchmark)} rows")
    print("Wrote data/processed/catem_benchmark_v1.csv")
    print("Wrote data/processed/catem_benchmark_v1_long.csv")
    print("Wrote outputs/catem_benchmark_validation.csv")


if __name__ == "__main__":
    main()
