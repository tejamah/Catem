import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.dataset_integrations import integrate_nasa_tlx_csv, save_integrated_dataset


def main() -> None:
    input_path = ROOT_DIR / "CATEM" / "datasets" / "nasa_tlx" / "nasa_tlx.csv"
    if not input_path.exists():
        print(f"Place a NASA-TLX CSV at {input_path}")
        return
    df = integrate_nasa_tlx_csv(input_path)
    save_integrated_dataset(df, "CATEM/outputs/nasa_tlx_catem.csv")
    print("Wrote CATEM/outputs/nasa_tlx_catem.csv")


if __name__ == "__main__":
    main()

