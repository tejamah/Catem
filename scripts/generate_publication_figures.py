from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "benchmark" / "catem_master.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def flow(title: str, labels: list[str], filename: str) -> None:
    plt.figure(figsize=(11, 3.8))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold")
    xs = np.linspace(0.07, 0.93, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.48,
            label,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f7fbff", "ec": "#0b2a66"},
        )
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.05, 0.48), xytext=(x + 0.05, 0.48), arrowprops={"arrowstyle": "->"})
    save(filename)


def correlation_figure(df: pd.DataFrame) -> None:
    cols = [
        "heart_rate",
        "hrv",
        "gsr",
        "mental_demand",
        "latency_ms",
        "tracking_error",
        "error_rate",
        "success_rate",
        "presence_score",
        "telepresence_quality",
    ]
    corr = df[[col for col in cols if col in df.columns]].corr()
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Figure 3 - Correlation Matrix", weight="bold")
    save("figure3_correlation_matrix.png")


def importance_figure() -> None:
    importance = pd.read_csv(TABLES / "feature_importance.csv").head(10).iloc[::-1]
    plt.figure(figsize=(7.2, 4.8))
    plt.barh(importance["feature"], importance["importance"], color="#6f3cc3")
    plt.xlabel("Importance")
    plt.title("Figure 4 - Feature Importance", weight="bold")
    save("figure4_feature_importance.png")


def model_comparison_figure() -> None:
    comparison = pd.read_csv(TABLES / "regression_model_comparison.csv")
    colors = ["#4f90d9" if model != "catem_full_model" else "#6f3cc3" for model in comparison["model"]]
    plt.figure(figsize=(8, 4.8))
    plt.bar(comparison["model"], comparison["r2"], color=colors)
    plt.ylabel("R2")
    plt.xticks(rotation=25, ha="right")
    plt.title("Figure 5 - Model Comparison", weight="bold")
    save("figure5_model_comparison.png")


def dashboard_overview(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title("Figure 6 - Dashboard Overview", fontsize=15, weight="bold")
    metrics = {
        "Records": len(df),
        "Participants": df["participant_id"].nunique(),
        "Tasks": df["task_id"].nunique(),
        "Target": "telepresence_quality",
    }
    for idx, (label, value) in enumerate(metrics.items()):
        ax.text(
            0.14 + idx * 0.24,
            0.68,
            f"{label}\n{value}",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.6", "fc": "#ffffff", "ec": "#9db4d6"},
        )
    trend = df["telepresence_quality"].ffill().fillna(df["telepresence_quality"].mean()).tail(80).to_numpy()
    ax.plot(np.linspace(0.08, 0.92, len(trend)), 0.12 + trend * 0.32, color="#5b35c8", lw=2)
    save("figure6_dashboard_overview.png")


def main() -> None:
    df = pd.read_csv(DATASET)
    flow(
        "Figure 1 - CATEM Architecture",
        ["Multimodal Data", "Layer Scores", "CATEM Model", "Validation", "Decision Support"],
        "figure1_catem_architecture.png",
    )
    flow(
        "Figure 2 - Data Pipeline",
        ["PhysioNet", "NASA-TLX", "RoboTurk", "ROS", "ETL", "catem_master.csv"],
        "figure2_data_pipeline.png",
    )
    correlation_figure(df)
    importance_figure()
    model_comparison_figure()
    dashboard_overview(df)
    print(f"Wrote publication figures to {FIGURES}")


if __name__ == "__main__":
    main()
