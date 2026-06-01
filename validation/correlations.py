from validation.common import load_master, numeric_features, write_table


def main() -> None:
    df = load_master()
    cols = numeric_features(df, exclude_target=False)
    correlations = df[cols].corr()["telepresence_quality"].sort_values(ascending=False).reset_index()
    correlations.columns = ["feature", "correlation_with_telepresence_quality"]
    path = write_table(correlations, "correlations.csv")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
