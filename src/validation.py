import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr()


def linear_regression_summary(df: pd.DataFrame, features: list[str], target: str) -> dict:
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    model = LinearRegression()
    model.fit(X, y)
    return {
        "coefficients": dict(zip(features, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "score": float(model.score(X, y)),
    }


def random_forest_importance(df: pd.DataFrame, features: list[str], target: str) -> dict:
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return {
        "importance": dict(zip(features, model.feature_importances_.tolist())),
        "r2": float(model.score(X_test, y_test)),
    }
