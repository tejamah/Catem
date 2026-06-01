import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from validation.common import MODEL_GROUPS, load_master, prepare_xy, write_table


def main() -> None:
    df = load_master()
    X, y = prepare_xy(df, MODEL_GROUPS["catem_full_model"])
    model = RandomForestRegressor(random_state=42, n_estimators=300)
    model.fit(X, y)
    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    path = write_table(importance, "feature_importance.csv")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
