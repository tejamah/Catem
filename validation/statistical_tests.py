from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from validation.common import MODEL_GROUPS, load_master, prepare_xy, write_table


RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 1000


def bootstrap_ci(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    x_values = x.reset_index(drop=True)
    y_values = y.reset_index(drop=True)
    n = len(x_values)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, n)
        sample_x = x_values.iloc[idx]
        sample_y = y_values.iloc[idx]
        if sample_x.nunique(dropna=True) < 2 or sample_y.nunique(dropna=True) < 2:
            continue
        if method == "spearman":
            values.append(spearmanr(sample_x, sample_y).statistic)
        else:
            values.append(pearsonr(sample_x, sample_y).statistic)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.percentile(values, [2.5, 97.5]))


def main() -> None:
    df = load_master()
    target = df["telepresence_quality"].fillna(df["telepresence_quality"].mean())
    rows = []
    absolute_errors = {}

    for model_name, features in MODEL_GROUPS.items():
        X, y = prepare_xy(df, features)
        if X.empty:
            continue
        model = LinearRegression()
        model.fit(X, y)
        prediction = pd.Series(model.predict(X), name=model_name)
        absolute_errors[model_name] = (prediction - target.reset_index(drop=True)).abs()

        pearson = pearsonr(prediction, target)
        spearman = spearmanr(prediction, target)
        pearson_low, pearson_high = bootstrap_ci(prediction, target, "pearson")
        spearman_low, spearman_high = bootstrap_ci(prediction, target, "spearman")
        rows.append(
            {
                "test": "prediction_vs_target",
                "model": model_name,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "pearson_ci_low": pearson_low,
                "pearson_ci_high": pearson_high,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
            }
        )

    if absolute_errors:
        groups = [errors.to_numpy() for errors in absolute_errors.values()]
        anova = f_oneway(*groups)
        rows.append(
            {
                "test": "anova_absolute_error_across_models",
                "model": "all_models",
                "pearson_r": np.nan,
                "pearson_p": anova.pvalue,
                "pearson_ci_low": np.nan,
                "pearson_ci_high": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": anova.pvalue,
                "spearman_ci_low": np.nan,
                "spearman_ci_high": np.nan,
            }
        )

    path = write_table(pd.DataFrame(rows), "statistical_tests.csv")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
