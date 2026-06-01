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


def selected_research_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return publication-facing cross-layer correlation pairs."""
    pairs = [
        ("ownership_score", "presence_score", "Ownership <-> Presence"),
        ("agency_score", "movement_smoothness", "Agency <-> Performance"),
        ("latency_ms", "agency_score", "Latency <-> Agency"),
        ("nasa_tlx_score", "error_rate", "Workload <-> Error Rate"),
    ]
    rows = []
    for left, right, label in pairs:
        if left in df.columns and right in df.columns:
            rows.append({"relationship": label, "correlation": float(df[left].corr(df[right]))})
    return pd.DataFrame(rows)


def top_predictors(df: pd.DataFrame, features: list[str], target: str = "catem_score", top_n: int = 5) -> pd.DataFrame:
    """Rank predictors for the CATEM target with a deterministic random forest."""
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    model = RandomForestRegressor(random_state=42, n_estimators=250)
    model.fit(X, y)
    ranked = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)
    return pd.DataFrame(ranked[:top_n], columns=["feature", "importance"])


def explain_score_drop(row: pd.Series, df: pd.DataFrame) -> list[str]:
    """Explain why a row underperformed relative to the sample distribution."""
    reasons = []
    checks = [
        ("latency_ms", "High latency", "high"),
        ("nasa_tlx_score", "High workload", "high"),
        ("tracking_loss", "Tracking instability", "high"),
        ("packet_loss", "Packet loss", "high"),
        ("agency_score", "Low agency", "low"),
        ("presence_score", "Low presence", "low"),
        ("hrv", "Reduced HRV", "low"),
    ]
    for col, label, direction in checks:
        if col not in df.columns or col not in row:
            continue
        if direction == "high" and row[col] >= df[col].quantile(0.70):
            reasons.append(label)
        if direction == "low" and row[col] <= df[col].quantile(0.30):
            reasons.append(label)
    return reasons[:4] or ["No major risk driver detected"]
