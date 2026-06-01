import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.data_loader import save_data


BENCHMARK_PATH = "CATEM/benchmark/catem_v1.csv"
OUTPUT_DIR = Path("CATEM/outputs")
MODEL_GROUPS = {
    "Presence": ["presence_score"],
    "Workload": ["nasa_tlx_total", "mental_demand", "effort", "frustration"],
    "System": ["latency_ms", "tracking_error", "packet_loss", "jitter", "fps"],
    "Performance": ["task_completion_time", "error_rate", "success_rate", "path_efficiency"],
    "CATEM": ["catem_score"],
}


def numeric_features(df: pd.DataFrame) -> list[str]:
    excluded = {
        "overall_telepresence_quality",
        "catem_score",
        "timestamp",
    }
    return [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in excluded and not df[col].dropna().empty
    ]


def main() -> None:
    df = pd.read_csv(BENCHMARK_PATH)
    target = df["overall_telepresence_quality"].fillna(df["overall_telepresence_quality"].mean())
    features = numeric_features(df)
    X = df[features].fillna(df[features].median(numeric_only=True)).fillna(0)

    comparison_rows = []
    for model_name, model_features in MODEL_GROUPS.items():
        available = [feature for feature in model_features if feature in df.columns and not df[feature].dropna().empty]
        if not available:
            continue
        model_X = df[available].fillna(df[available].median(numeric_only=True)).fillna(0)
        model = LinearRegression()
        model.fit(model_X, target)
        prediction = model.predict(model_X)
        comparison_rows.append(
            {
                "model": model_name,
                "r2": model.score(model_X, target),
                "mae": mean_absolute_error(target, prediction),
                "rmse": mean_squared_error(target, prediction) ** 0.5,
                "correlation": pd.Series(prediction).corr(target.reset_index(drop=True)),
                "feature_count": len(available),
            }
        )
    save_data(pd.DataFrame(comparison_rows), OUTPUT_DIR / "catem_v1_model_comparison.csv")

    correlations = df[features + ["catem_score", "overall_telepresence_quality"]].corr()
    save_data(correlations, OUTPUT_DIR / "catem_v1_correlations.csv")

    regression = LinearRegression()
    regression.fit(X, target)
    regression_rows = pd.DataFrame(
        {
            "feature": features,
            "coefficient": regression.coef_,
        }
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    regression_rows.loc[len(regression_rows)] = ["intercept", regression.intercept_]
    regression_rows.loc[len(regression_rows)] = ["r2", regression.score(X, target)]
    save_data(regression_rows, OUTPUT_DIR / "catem_v1_regression.csv")

    forest = RandomForestRegressor(random_state=42, n_estimators=300)
    forest.fit(X, target)
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    save_data(importance, OUTPUT_DIR / "catem_v1_feature_importance.csv")

    print("Wrote CATEM/outputs/catem_v1_correlations.csv")
    print("Wrote CATEM/outputs/catem_v1_regression.csv")
    print("Wrote CATEM/outputs/catem_v1_feature_importance.csv")
    print("Wrote CATEM/outputs/catem_v1_model_comparison.csv")


if __name__ == "__main__":
    main()
