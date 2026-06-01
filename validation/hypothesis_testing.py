import pandas as pd
from scipy.stats import pearsonr

from validation.common import MODEL_GROUPS, load_master, prepare_xy, write_table


def main() -> None:
    df = load_master()
    target = df["telepresence_quality"].fillna(df["telepresence_quality"].mean())
    rows = []
    for model_name, features in MODEL_GROUPS.items():
        X, _ = prepare_xy(df, features)
        if X.empty:
            continue
        predictor = X.mean(axis=1)
        statistic, p_value = pearsonr(predictor, target)
        rows.append(
            {
                "hypothesis": f"{model_name} predicts telepresence_quality",
                "pearson_r": statistic,
                "p_value": p_value,
                "supported_at_0_05": p_value < 0.05,
            }
        )
    path = write_table(pd.DataFrame(rows), "hypothesis_tests.csv")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
