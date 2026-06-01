import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

BENCHMARK_PATH = ROOT_DIR / "CATEM" / "benchmark" / "catem_v1.csv"
OUTPUT_DIR = ROOT_DIR / "CATEM" / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def save_figure(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def draw_flow_figure(title: str, labels: list[str], filename: str) -> None:
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=16)
    x_positions = np.linspace(0.06, 0.94, len(labels))
    for idx, (x_pos, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x_pos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.55", "fc": "#f8fbff", "ec": "#17417d", "lw": 1.5},
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.055, 0.5),
                xytext=(x_pos + 0.055, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#17417d"},
            )
    save_figure(filename)


def figure_schema(df: pd.DataFrame) -> None:
    groups = {
        "Identifiers": ["participant_id", "session_id", "task_id", "timestamp"],
        "Physiology": ["heart_rate", "hrv", "ecg_signal", "gsr", "eda", "skin_temp"],
        "Workload": ["mental_demand", "physical_demand", "temporal_demand", "effort", "frustration", "nasa_tlx_total"],
        "System": ["latency_ms", "tracking_error", "packet_loss", "jitter", "fps"],
        "Performance": ["task_completion_time", "error_rate", "success_rate", "path_efficiency"],
        "Human State": ["ownership_score", "agency_score", "presence_score"],
        "CATEM Outputs": ["overall_telepresence_quality", "physiology_score", "workload_score", "system_score", "performance_score", "catem_score"],
    }
    plt.figure(figsize=(13, 6))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title("Figure 3. CATEM Benchmark Dataset Schema", fontsize=16, weight="bold", pad=16)
    y = 0.92
    for group, columns in groups.items():
        available = [col for col in columns if col in df.columns]
        ax.text(0.02, y, group, fontsize=11, weight="bold", color="#08245a")
        ax.text(0.22, y, ", ".join(available), fontsize=9)
        y -= 0.12
    ax.text(0.02, 0.04, f"Records: {len(df):,} | Columns: {len(df.columns):,}", fontsize=11, weight="bold")
    save_figure("figure3_dataset_schema.png")


def figure_correlation(df: pd.DataFrame) -> None:
    cols = [
        "ownership_score",
        "agency_score",
        "presence_score",
        "nasa_tlx_total",
        "latency_ms",
        "error_rate",
        "success_rate",
        "catem_score",
        "overall_telepresence_quality",
    ]
    corr = df[[col for col in cols if col in df.columns]].corr()
    plt.figure(figsize=(8, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
    plt.title("Figure 4. CATEM Correlation Matrix", fontsize=15, weight="bold")
    save_figure("figure4_correlation_matrix.png")


def figure_importance() -> None:
    path = OUTPUT_DIR / "catem_v1_feature_importance.csv"
    if not path.exists():
        return
    importance = pd.read_csv(path).head(10).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(importance["feature"], importance["importance"], color="#6f3cc3")
    plt.xlabel("Importance")
    plt.title("Figure 5. Feature Importance", fontsize=15, weight="bold")
    save_figure("figure5_feature_importance.png")


def figure_model_comparison() -> None:
    path = OUTPUT_DIR / "catem_v1_model_comparison.csv"
    if not path.exists():
        return
    comparison = pd.read_csv(path)
    plt.figure(figsize=(8, 5))
    plt.bar(comparison["model"], comparison["r2"], color=["#5d9bd8", "#5d9bd8", "#5d9bd8", "#5d9bd8", "#6f3cc3"])
    plt.ylabel("R2")
    plt.ylim(0, max(1.0, float(comparison["r2"].max()) + 0.1))
    plt.title("Figure 6. Model Comparison", fontsize=15, weight="bold")
    save_figure("figure6_model_comparison.png")


def figure_dashboard_overview(df: pd.DataFrame) -> None:
    latest = df.tail(1).iloc[0]
    cards = [
        ("CATEM", latest.get("catem_score", np.nan)),
        ("Physiology", latest.get("physiology_score", np.nan)),
        ("Workload", latest.get("workload_score", np.nan)),
        ("System", latest.get("system_score", np.nan)),
        ("Performance", latest.get("performance_score", np.nan)),
    ]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axis("off")
    ax.set_title("Figure 7. CATEM Dashboard Overview", fontsize=16, weight="bold", pad=14)
    for idx, (label, value) in enumerate(cards):
        x = 0.07 + idx * 0.185
        ax.text(
            x,
            0.72,
            f"{label}\n{float(value) * 100:0.1f}/100" if pd.notna(value) else f"{label}\nN/A",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.65", "fc": "#ffffff", "ec": "#b7c7df", "lw": 1.4},
        )
    trend = df["catem_score"].ffill().fillna(0.5).tail(50).to_numpy()
    ax.plot(np.linspace(0.08, 0.92, len(trend)), 0.2 + trend * 0.35, color="#5b35c8", lw=2.5)
    ax.fill_between(np.linspace(0.08, 0.92, len(trend)), 0.2, 0.2 + trend * 0.35, color="#d8cff6", alpha=0.6)
    ax.text(0.08, 0.12, "Score trend, validation models, correlations, explainability, and recommendations", fontsize=11)
    save_figure("figure7_dashboard_overview.png")


def main() -> None:
    df = pd.read_csv(BENCHMARK_PATH)
    draw_flow_figure(
        "Figure 1. CATEM Architecture",
        ["Embodiment", "Presence", "Behavior", "Physiology", "Workload", "System", "Data Quality", "CATEM Score"],
        "figure1_catem_architecture.png",
    )
    draw_flow_figure(
        "Figure 2. CATEM Data Pipeline",
        ["Public Data", "ETL", "Feature Extraction", "Layer Scores", "Validation", "Dashboard", "Decision Support"],
        "figure2_data_pipeline.png",
    )
    figure_schema(df)
    figure_correlation(df)
    figure_importance()
    figure_model_comparison()
    figure_dashboard_overview(df)
    print(f"Wrote publication figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
