import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from validation.common import MODEL_GROUPS, load_master, prepare_xy, write_table


def main() -> None:
    df = load_master()
    rows = []
    for model_name, features in MODEL_GROUPS.items():
        X, y = prepare_xy(df, features)
        if X.empty:
            continue
        model = RandomForestRegressor(random_state=42, n_estimators=250)
        model.fit(X, y)
        prediction = model.predict(X)
        rows.append(
            {
                "model": model_name,
                "r2": model.score(X, y),
                "mae": mean_absolute_error(y, prediction),
                "rmse": mean_squared_error(y, prediction) ** 0.5,
                "correlation": pd.Series(prediction).corr(y.reset_index(drop=True)),
                "feature_count": X.shape[1],
            }
        )
    path = write_table(pd.DataFrame(rows), "random_forest_model_comparison.csv")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
