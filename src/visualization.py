import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str], title: str = "Correlation Heatmap") -> plt.Figure:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    return fig


def plot_layer_scores(df: pd.DataFrame, participant_id: str) -> plt.Figure:
    layers = [
        "embodiment_score",
        "presence_score",
        "behavior_score",
        "physiology_score",
        "system_stability_score",
        "data_quality_score",
        "workload_score",
    ]
    subset = df[df["participant_id"] == participant_id]
    if subset.empty:
        raise ValueError(f"No data for participant {participant_id}")
    fig, ax = plt.subplots(figsize=(10, 6))
    subset[layers].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Score")
    ax.set_title(f"CATEM Layer Averages for {participant_id}")
    ax.set_ylim(0, 1)
    return fig
