import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.benchmark_dataset import build_benchmark_from_user_files


DOWNLOADS = Path.home() / "Downloads"
ROBOTURK_RESULTS = DOWNLOADS / "roboturk_real_dataset-master" / "roboturk_real_dataset-master" / "results"


def main() -> None:
    benchmark = build_benchmark_from_user_files(
        demographics_path=DOWNLOADS / "Demographics.csv",
        tlx_assistant_path=DOWNLOADS / "nasatlx_assistant.csv",
        tlx_current_path=DOWNLOADS / "nasatlx_current.csv",
        roboturk_results_dir=ROBOTURK_RESULTS,
        zenodo_workbook_path=DOWNLOADS / "dataset-main" / "concatenated_data_zenodo.xlsx",
        robot_anomaly_path=DOWNLOADS / "robot_anomaly_full.csv",
    )
    print(f"Built CATEM user benchmark with {len(benchmark)} rows")
    print("Wrote data/processed/catem_benchmark_user_data.csv")
    print("Wrote data/processed/catem_benchmark_user_data_long.csv")
    print("Wrote outputs/catem_benchmark_user_validation.csv")


if __name__ == "__main__":
    main()
