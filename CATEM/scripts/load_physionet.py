import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.dataset_integrations import integrate_physionet_wearable_csv, save_integrated_dataset


def main() -> None:
    input_path = ROOT_DIR / "CATEM" / "datasets" / "physionet" / "physionet_wearable.csv"
    if not input_path.exists():
        print(f"Place a PhysioNet/WESAD-derived CSV at {input_path}")
        return
    df = integrate_physionet_wearable_csv(input_path, participant_id="physionet", session_id="S01")
    save_integrated_dataset(df, "CATEM/outputs/physionet_catem.csv")
    print("Wrote CATEM/outputs/physionet_catem.csv")


if __name__ == "__main__":
    main()

