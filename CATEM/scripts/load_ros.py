import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.dataset_integrations import integrate_ros_telemetry_csv, save_integrated_dataset


def main() -> None:
    input_path = ROOT_DIR / "CATEM" / "datasets" / "rosbags" / "ros_telemetry.csv"
    if not input_path.exists():
        print(f"Export ROS bag telemetry to CSV and place it at {input_path}")
        return
    df = integrate_ros_telemetry_csv(input_path, participant_id="ros", session_id="S01")
    save_integrated_dataset(df, "CATEM/outputs/ros_catem.csv")
    print("Wrote CATEM/outputs/ros_catem.csv")


if __name__ == "__main__":
    main()

