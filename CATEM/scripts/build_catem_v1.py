import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
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
        apple_health_export_path=DOWNLOADS / "apple_health_export" / "export.xml",
        output_path="CATEM/benchmark/catem_v1.csv",
        long_output_path="CATEM/outputs/catem_v1_long.csv",
        validation_output_path="CATEM/outputs/catem_v1_validation.csv",
    )
    print(f"Built CATEM Dataset v1 with {len(benchmark)} records")
    print("Wrote CATEM/benchmark/catem_v1.csv")
    print("Wrote CATEM/outputs/catem_v1_long.csv")
    print("Wrote CATEM/outputs/catem_v1_validation.csv")


if __name__ == "__main__":
    main()
