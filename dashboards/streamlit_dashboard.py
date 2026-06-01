import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.catem_scoring import compute_catem_scores
from src.data_loader import load_data
from src.validation import explain_score_drop, selected_research_correlations, top_predictors


LAYER_META = [
    {
        "number": "1",
        "name": "EMBODIMENT LAYER",
        "color": "#6f3cc3",
        "glyph": "E",
        "metrics": ["Ownership", "Agency", "Self-Location"],
        "description": "Sense of body ownership and control",
    },
    {
        "number": "2",
        "name": "PRESENCE LAYER",
        "color": "#2167c8",
        "glyph": "P",
        "metrics": ["Spatial Presence", "Social Presence"],
        "description": 'Sense of "being there" in the remote environment',
    },
    {
        "number": "3",
        "name": "BEHAVIOR LAYER",
        "color": "#31a14d",
        "glyph": "B",
        "metrics": ["Task Performance", "Interaction Behavior", "Movement Metrics"],
        "description": "Objective behavior and performance",
    },
    {
        "number": "4",
        "name": "PHYSIOLOGICAL LAYER",
        "color": "#f06423",
        "glyph": "H",
        "metrics": ["Heart Rate, HRV", "GSR, EEG, Eye Tracking", "Blink Rate, Pupil Dilation"],
        "description": "Physiological responses and engagement",
    },
    {
        "number": "5",
        "name": "WORKLOAD LAYER",
        "color": "#f5a51d",
        "glyph": "W",
        "metrics": ["NASA-TLX", "Mental Demand, Effort", "Frustration, Temporal Demand"],
        "description": "Cognitive workload and mental resources",
    },
    {
        "number": "6",
        "name": "SYSTEM LAYER",
        "color": "#22a8b9",
        "glyph": "S",
        "metrics": ["Latency, Jitter", "FPS, Packet Loss", "Tracking Quality"],
        "description": "System performance and technical stability",
    },
    {
        "number": "7",
        "name": "DATA QUALITY LAYER",
        "color": "#6c7888",
        "glyph": "D",
        "metrics": ["Missing Data Rate", "Timestamp Accuracy", "Sensor Synchronization"],
        "description": "Data reliability and measurement quality",
    },
]


@st.cache_data
def load_and_score() -> pd.DataFrame:
    df = load_data("data/synthetic/catem_sample_data.csv")
    return compute_catem_scores(df)


def scaled(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 100).round(1)


def score_value(df: pd.DataFrame, col: str) -> float:
    return float(scaled(df[col]).mean())


def status_for(value: float) -> tuple[str, str]:
    if value >= 85:
        return "Excellent", "#00843d"
    if value >= 70:
        return "Good", "#00843d"
    if value >= 60:
        return "Moderate", "#e9571b"
    return "Watch", "#b00020"


def sparkline(values: pd.Series, color: str = "#1d6ee8") -> str:
    nums = values.astype(float).to_numpy()
    if nums.max() == nums.min():
        ys = np.full_like(nums, 16.0)
    else:
        ys = 26 - ((nums - nums.min()) / (nums.max() - nums.min()) * 18)
    xs = np.linspace(4, 96, len(nums))
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f"<svg class='spark' viewBox='0 0 100 32' preserveAspectRatio='none'>"
        f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.2' "
        "stroke-linecap='round' stroke-linejoin='round'/>"
        "</svg>"
    )


def metric_card(title: str, value: float, denominator: str = "/100", color: str = "#1d6ee8") -> str:
    status, status_color = status_for(value)
    spark = sparkline(pd.Series([value - 8, value - 5, value - 7, value - 2, value - 4, value + 1, value - 3, value + 2]), color)
    return (
        '<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div><span class="metric-value">{value:.1f}</span> <span class="metric-denom">{denominator}</span></div>'
        f'<div class="metric-status" style="color:{status_color};">{status}</div>'
        f"{spark}"
        "</div>"
    )


def framework_layer(meta: dict) -> str:
    metrics = "".join(f"<li>{item}</li>" for item in meta["metrics"])
    return (
        f'<div class="layer-row" style="border-color:{meta["color"]}55;">'
        f'<div class="layer-icon" style="background:{meta["color"]};">{meta["glyph"]}</div>'
        '<div class="layer-main">'
        f'<div class="layer-title" style="color:{meta["color"]};">{meta["number"]}. {meta["name"]}</div>'
        f"<ul>{metrics}</ul>"
        "</div>"
        f'<div class="layer-desc">{meta["description"]}</div>'
        "</div>"
    )


def section_header(title: str) -> str:
    return f"<div class='section-title'>{title}</div>"


def build_trend(df: pd.DataFrame, column: str, points: int = 31, noise: float = 4.0) -> pd.DataFrame:
    base = scaled(df[column]).to_numpy()
    x_old = np.linspace(0, points - 1, len(base))
    x_new = np.arange(points)
    values = np.interp(x_new, x_old, base)
    wave = np.sin(np.linspace(0, 5.5 * np.pi, points)) * noise
    return pd.DataFrame({"minute": x_new, "value": np.clip(values + wave, 0, 100)})


def line_chart(df: pd.DataFrame) -> go.Figure:
    trend = build_trend(df, "catem_score", noise=5.5)
    trend["value"] = np.clip(trend["value"] / max(trend["value"].max(), 1) * 78, 45, 86)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["minute"],
            y=trend["value"],
            mode="lines+markers",
            line=dict(color="#5d2bc5", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(93,43,197,0.10)",
            name="CATEM",
        )
    )
    fig.update_layout(
        title=dict(text="CATEM Score Trend Over Time", x=0.5, font=dict(size=14, color="#111827")),
        height=245,
        margin=dict(l=36, r=18, t=42, b=34),
        xaxis_title="Time (min)",
        yaxis=dict(range=[0, 100], title="", gridcolor="#e7ecf5"),
        xaxis=dict(tickvals=[0, 5, 10, 15, 20, 25, 30], ticktext=["00:00", "05:00", "10:00", "15:00", "20:00", "25:00", "30:00"], gridcolor="#eef2f7"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(size=10, color="#111827"),
    )
    return fig


def radar_chart(df: pd.DataFrame) -> go.Figure:
    categories = ["Embodiment", "Presence", "Behavior", "Physiology", "Workload (-)", "System", "Data Quality"]
    values = [
        score_value(df, "embodiment_score"),
        score_value(df, "presence_score"),
        score_value(df, "behavior_score"),
        score_value(df, "physiology_score"),
        score_value(df, "workload_score"),
        score_value(df, "system_stability_score"),
        score_value(df, "data_quality_score"),
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#5d2bc5", width=2),
            fillcolor="rgba(93,43,197,0.16)",
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title=dict(text="Layer Contribution to CATEM Score", x=0.5, font=dict(size=14, color="#111827")),
        height=245,
        margin=dict(l=30, r=30, t=42, b=20),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8), gridcolor="#dde6f2"),
            angularaxis=dict(tickfont=dict(size=9, color="#111827"), gridcolor="#dde6f2"),
        ),
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=10, color="#111827"),
    )
    return fig


def score_trend_trace(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    trend = build_trend(df, "catem_score", noise=5.5)
    trend["value"] = np.clip(trend["value"] / max(trend["value"].max(), 1) * 78, 45, 86)
    return trend["minute"], trend["value"]


def top_charts(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=("CATEM Trend", "Layer Contribution"),
        horizontal_spacing=0.12,
    )
    minutes, values = score_trend_trace(df)
    fig.add_trace(
        go.Scatter(
            x=minutes,
            y=values,
            mode="lines+markers",
            line=dict(color="#5d2bc5", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(93,43,197,0.10)",
            name="CATEM",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    categories = ["Embodiment", "Presence", "Behavior", "Physiology", "Workload (-)", "System", "Data Quality"]
    radar_values = [
        score_value(df, "embodiment_score"),
        score_value(df, "presence_score"),
        score_value(df, "behavior_score"),
        score_value(df, "physiology_score"),
        score_value(df, "workload_score"),
        score_value(df, "system_stability_score"),
        score_value(df, "data_quality_score"),
    ]
    fig.add_trace(
        go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color="#5d2bc5", width=2),
            fillcolor="rgba(93,43,197,0.16)",
            marker=dict(size=4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=20, t=34, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10, color="#111827"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=7), gridcolor="#dde6f2"),
            angularaxis=dict(tickfont=dict(size=8), gridcolor="#dde6f2"),
        ),
    )
    fig.update_annotations(font_size=13)
    fig.update_xaxes(
        title_text="",
        tickvals=[0, 5, 10, 15, 20, 25, 30],
        ticktext=["00:00", "05:00", "10:00", "15:00", "20:00", "25:00", "30:00"],
        gridcolor="#eef2f7",
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 100], gridcolor="#e7ecf5", row=1, col=1)
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker=dict(color=color, size=8, opacity=0.86),
            name="Session",
        )
    )
    if len(df) > 1:
        z = np.polyfit(df[x].astype(float), df[y].astype(float), 1)
        xs = np.linspace(df[x].min(), df[x].max(), 40)
        fig.add_trace(go.Scatter(x=xs, y=np.poly1d(z)(xs), mode="lines", line=dict(color=color, width=2), showlegend=False))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=12, color="#111827")),
        height=200,
        margin=dict(l=42, r=12, t=34, b=40),
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(gridcolor="#eef2f7"),
        yaxis=dict(gridcolor="#eef2f7"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        font=dict(size=9, color="#111827"),
    )
    return fig


def scatter_charts(df: pd.DataFrame) -> go.Figure:
    specs = [
        ("latency_ms", "agency_score", "Latency vs Agency", "#5d2bc5", "Latency", "Agency"),
        ("nasa_tlx_score", "task_completion_time", "Workload vs Task", "#ff6b1a", "Workload", "Time"),
        ("tracking_loss", "gsr", "Tracking vs GSR", "#188b3b", "Tracking", "GSR"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[item[2] for item in specs], horizontal_spacing=0.09)
    for idx, (x, y, _title, color, x_label, y_label) in enumerate(specs, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="markers",
                marker=dict(color=color, size=8, opacity=0.86),
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        if len(df) > 1:
            z = np.polyfit(df[x].astype(float), df[y].astype(float), 1)
            xs = np.linspace(df[x].min(), df[x].max(), 40)
            fig.add_trace(
                go.Scatter(x=xs, y=np.poly1d(z)(xs), mode="lines", line=dict(color=color, width=2), showlegend=False),
                row=1,
                col=idx,
            )
        fig.update_xaxes(title_text=x_label, gridcolor="#eef2f7", row=1, col=idx)
        fig.update_yaxes(title_text=y_label, gridcolor="#eef2f7", row=1, col=idx)
    fig.update_layout(
        height=185,
        margin=dict(l=34, r=10, t=34, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=9, color="#111827"),
    )
    fig.update_annotations(font_size=12)
    return fig


def telemetry_chart(df: pd.DataFrame) -> go.Figure:
    minutes = np.arange(31)
    latency = np.interp(minutes, np.linspace(0, 30, len(df)), df["latency_ms"]) + np.sin(minutes * 0.9) * 8
    fps = np.interp(minutes, np.linspace(0, 30, len(df)), df["fps"]) + np.cos(minutes * 0.7) * 3
    packet_loss = np.interp(minutes, np.linspace(0, 30, len(df)), df["packet_loss"] * 100) + np.sin(minutes * 1.4) * 0.5
    tracking = np.interp(minutes, np.linspace(0, 30, len(df)), df["tracking_loss"] * 100) + np.cos(minutes * 1.2) * 0.4
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=minutes, y=latency, mode="lines+markers", name="Latency (ms)", line=dict(color="#5d2bc5", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=minutes, y=fps, mode="lines+markers", name="FPS", line=dict(color="#1d6ee8", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=minutes, y=packet_loss, mode="lines+markers", name="Packet Loss (%)", line=dict(color="#ff6b1a", width=2), marker=dict(size=4), yaxis="y2"))
    fig.add_trace(go.Scatter(x=minutes, y=tracking, mode="lines+markers", name="Tracking Loss", line=dict(color="#0b8f3a", width=2), marker=dict(size=4), yaxis="y2"))
    fig.update_layout(
        title=dict(text="System Telemetry Over Time", x=0.5, font=dict(size=13, color="#111827")),
        height=165,
        margin=dict(l=36, r=36, t=32, b=24),
        xaxis_title="",
        yaxis=dict(title="", gridcolor="#eef2f7"),
        yaxis2=dict(title="", overlaying="y", side="right", range=[0, 10]),
        legend=dict(orientation="h", y=1.22, x=0.22, font=dict(size=8)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=9, color="#111827"),
    )
    return fig


def comparison_chart() -> go.Figure:
    labels = ["Presence Only", "Embodiment Only", "Workload Only", "Performance Only", "System Telemetry Only", "CATEM (All Layers)"]
    vals = [0.42, 0.49, 0.36, 0.55, 0.48, 0.78]
    colors = ["#5f9bd8", "#5f9bd8", "#5f9bd8", "#5f9bd8", "#5f9bd8", "#6f3cc3"]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors, text=vals, textposition="outside"))
    fig.update_layout(
        height=150,
        margin=dict(l=108, r=24, t=4, b=18),
        xaxis=dict(range=[0, 1.0], title="", gridcolor="#eef2f7", tickfont=dict(size=8)),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=9, color="#111827"),
        showlegend=False,
    )
    return fig


def correlation_chart(df: pd.DataFrame) -> go.Figure:
    cols = ["ownership_score", "presence_score", "agency_score", "movement_smoothness", "latency_ms", "nasa_tlx_score", "error_rate"]
    labels = ["Ownership", "Presence", "Agency", "Performance", "Latency", "Workload", "Error Rate"]
    corr = df[cols].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(thickness=8),
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=62, r=8, t=4, b=36),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        font=dict(size=8, color="#111827"),
    )
    return fig


def feature_importance_chart(df: pd.DataFrame) -> go.Figure:
    features = [
        "agency_score",
        "presence_score",
        "latency_ms",
        "nasa_tlx_score",
        "hrv",
        "movement_smoothness",
        "tracking_loss",
        "error_rate",
    ]
    ranking = top_predictors(df, features)
    fig = go.Figure(
        go.Bar(
            x=ranking["importance"],
            y=ranking["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color="#6f3cc3",
            text=ranking["importance"].round(2),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=155,
        margin=dict(l=108, r=22, t=4, b=18),
        xaxis=dict(title="", range=[0, max(0.2, float(ranking["importance"].max()) * 1.25)], gridcolor="#eef2f7", tickfont=dict(size=8)),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=9, color="#111827"),
        showlegend=False,
    )
    return fig


def explainability_card(df: pd.DataFrame) -> str:
    lowest = df.sort_values("catem_score").iloc[0]
    reasons = explain_score_drop(lowest, df)
    reason_items = "".join(f"<li>{reason}</li>" for reason in reasons)
    return (
        '<div class="info-card insight">'
        "<h4>Explainability</h4>"
        f"<b>Why did {lowest['participant_id']} score lower?</b>"
        f"<ul>{reason_items}</ul>"
        "</div>"
    )


def nav_panel() -> str:
    items = [
        ("Overview", True),
        ("Layer Scores", False),
        ("Participant Analysis", False),
        ("Task Performance", False),
        ("Physiology & Workload", False),
        ("System Telemetry", False),
        ("Correlation Analysis", False),
        ("Feature Importance", False),
        ("Explainability", False),
        ("About CATEM", False),
    ]
    rows = "".join(f"<div class='nav-item {'active' if active else ''}'><span></span>{name}</div>" for name, active in items)
    return (
        '<div class="inner-nav">'
        '<div class="nav-title">CATEM Dashboard</div>'
        f"{rows}"
        '<div class="filter-box">'
        '<div class="filter-title">Filters</div>'
        '<label>Participant</label><div class="select-like">All</div>'
        '<label>Task</label><div class="select-like">All</div>'
        '<label>Session</label><div class="select-like">All</div>'
        '<label>Date Range</label><div class="date-row"><span>2024-01-01</span><span>to</span><span>2024-12-31</span></div>'
        "<button>Apply Filters</button>"
        "</div>"
        "</div>"
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #071c49;
            --navy2: #001641;
            --border: #b9c7dc;
            --soft: #f7fbff;
            --text: #081537;
            --blue: #1d6ee8;
        }
        #MainMenu, header, footer, [data-testid="stSidebar"] {display: none;}
        .block-container {
            max-width: 1880px;
            padding: 3.1rem 0.45rem 0.15rem;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #eef4fb;
            color: var(--text);
        }
        div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
        div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
        [data-testid="stPlotlyChart"] {
            overflow: hidden;
        }
        .hero {
            color: white;
            background: radial-gradient(circle at 18% 12%, #163c78 0%, #071f50 38%, #04163b 100%);
            border: 1px solid #0d3572;
            border-radius: 8px;
            text-align: center;
            padding: 7px 18px 8px;
            box-shadow: inset 0 0 26px rgba(255,255,255,0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: 36px;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: 0;
        }
        .hero .subtitle {
            color: #ffd232;
            font-size: 18px;
            font-style: italic;
            font-weight: 800;
            margin-top: 2px;
        }
        .hero-strip {
            margin: 7px auto 0;
            display: flex;
            justify-content: center;
            gap: 52px;
            font-weight: 700;
            font-size: 13px;
        }
        .hero-chip {
            display: flex;
            align-items: center;
            gap: 10px;
            white-space: nowrap;
        }
        .hero-icon {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(255,255,255,0.18);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            color: #ffffff;
        }
        .panel {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
            border: 1px solid var(--border);
            border-radius: 7px;
            overflow: hidden;
        }
        .section-title {
            background: linear-gradient(180deg, #102b6e 0%, #001a57 100%);
            color: white;
            text-align: center;
            font-weight: 900;
            font-size: 17px;
            line-height: 1.1;
            padding: 7px 8px;
            border-radius: 6px 6px 0 0;
        }
        .panel-body {
            padding: 10px 12px;
        }
        .framework-intro {
            text-align: center;
            font-size: 13px;
            line-height: 1.25;
            margin: 0 0 9px;
        }
        .layer-row {
            display: grid;
            grid-template-columns: 66px 1fr 116px;
            min-height: 70px;
            border: 1px solid;
            border-radius: 8px;
            margin-bottom: 7px;
            overflow: hidden;
            background: #ffffff;
        }
        .layer-icon {
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            font-weight: 900;
        }
        .layer-main {
            padding: 8px 7px 7px 12px;
            border-right: 1px solid #d6dfeb;
        }
        .layer-title {
            font-size: 14px;
            font-weight: 900;
            margin-bottom: 3px;
        }
        .layer-main ul {
            margin: 0;
            padding-left: 14px;
            font-size: 10.5px;
            line-height: 1.22;
            color: #000;
        }
        .layer-desc {
            display: flex;
            align-items: center;
            padding: 7px 10px;
            font-size: 10.5px;
            line-height: 1.25;
            color: #000;
        }
        .formula {
            border: 1px solid var(--border);
            border-radius: 5px;
            text-align: center;
            overflow: hidden;
            margin-top: 8px;
            font-size: 14px;
            background: white;
        }
        .formula-title {
            background: var(--navy);
            color: white;
            font-weight: 900;
            padding: 5px;
        }
        .formula-body {
            padding: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 7px;
            margin-bottom: 8px;
        }
        .metric-card {
            min-height: 102px;
            border: 1px solid #c9d4e5;
            border-radius: 7px;
            background: white;
            text-align: center;
            padding: 9px 6px 5px;
        }
        .metric-title {
            min-height: 28px;
            font-weight: 900;
            color: #001b61;
            font-size: 11px;
            line-height: 1.15;
        }
        .metric-value {
            color: #07123a;
            font-size: 25px;
            font-weight: 900;
        }
        .metric-denom {
            color: #001;
            font-size: 13px;
            font-weight: 800;
        }
        .metric-status {
            font-size: 12px;
            font-weight: 800;
            margin-top: -2px;
        }
        .spark {
            width: 88%;
            height: 22px;
            margin-top: 2px;
        }
        .inner-nav {
            border-right: 1px solid #cfd9e8;
            min-height: 0;
            padding: 12px 10px 10px;
            background: linear-gradient(180deg, #ffffff 0%, #f6f9fd 100%);
        }
        .nav-title {
            font-size: 16px;
            font-weight: 900;
            margin-bottom: 10px;
        }
        .nav-item {
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 6px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 3px;
        }
        .nav-item span {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            border: 1px solid #26436c;
        }
        .nav-item.active {
            background: linear-gradient(90deg, #1f7bf0, #1265d5);
            color: white;
        }
        .nav-item.active span {
            border-color: white;
        }
        .filter-box {
            border-top: 1px solid #cfd9e8;
            margin-top: 10px;
            padding-top: 8px;
        }
        .filter-title, .filter-box label {
            display: block;
            font-size: 11px;
            font-weight: 900;
            margin: 8px 0 4px;
        }
        .select-like, .date-row span {
            border: 1px solid #cbd6e6;
            border-radius: 4px;
            padding: 6px;
            background: white;
            font-size: 10px;
        }
        .date-row {
            display: grid;
            grid-template-columns: 1fr 20px 1fr;
            gap: 3px;
            align-items: center;
            text-align: center;
            font-size: 10px;
        }
        .filter-box button {
            width: 100%;
            border: 0;
            border-radius: 4px;
            margin-top: 10px;
            background: #1d6ee8;
            color: white;
            font-size: 11px;
            padding: 7px;
            font-weight: 800;
        }
        .chart-card {
            border: 1px solid #c9d4e5;
            border-radius: 7px;
            background: white;
            padding: 3px 3px 0;
            overflow: hidden;
        }
        .info-card {
            border: 1px solid #c9d4e5;
            border-radius: 7px;
            background: white;
            padding: 9px 12px;
            margin-bottom: 8px;
            font-size: 12px;
            line-height: 1.28;
        }
        .info-card h4 {
            color: #001b61;
            font-size: 13px;
            margin: 0 0 5px;
            font-weight: 900;
        }
        .info-card ul {
            margin: 5px 0 0;
            padding-left: 18px;
        }
        .insight {
            background: #eefaf1;
            border-color: #b7dec4;
        }
        @media (max-width: 1200px) {
            .hero h1 {font-size: 30px;}
            .hero-strip {gap: 18px; flex-wrap: wrap;}
            .metric-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
            .layer-row {grid-template-columns: 54px 1fr;}
            .layer-desc {grid-column: 1 / -1; min-height: 40px; border-top: 1px solid #d6dfeb;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="CATEM Telepresence Dashboard", layout="wide")
    apply_styles()
    df = load_and_score()

    st.markdown(
        """
        <div class="hero">
            <h1>CATEM - Cross-Layer Adaptive Telepresence Evaluation Model</h1>
            <div class="subtitle">A Multi-Dimensional Framework for Evaluating Telepresence Systems</div>
            <div class="hero-strip">
                <div class="hero-chip"><span class="hero-icon">H</span> Integrates Human, System & Environmental Factors</div>
                <div class="hero-chip"><span class="hero-icon">D</span> Uses Multimodal Data</div>
                <div class="hero-chip"><span class="hero-icon">Q</span> Generates Unified Telepresence Quality Score</div>
                <div class="hero-chip"><span class="hero-icon">A</span> Supports Analysis, Visualization & Validation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([0.245, 0.56, 0.225], gap="small")

    with left:
        st.markdown(f"<div class='panel'>{section_header('CATEM FRAMEWORK')}<div class='panel-body'>", unsafe_allow_html=True)
        st.markdown(
            "<p class='framework-intro'>CATEM evaluates telepresence quality<br>by integrating multiple cross-layer factors<br>into a single adaptive score.</p>",
            unsafe_allow_html=True,
        )
        for meta in LAYER_META:
            st.markdown(framework_layer(meta), unsafe_allow_html=True)
        st.markdown(
            """
            <div class="formula">
                <div class="formula-title">LITERATURE-WEIGHTED CATEM SCORE</div>
                <div class="formula-body">
                    0.25 Embodiment + 0.20 Presence + 0.20 Behavior<br>
                    + 0.10 Physiology + 0.15 System + 0.10 Data Quality<br>
                    <b style="color:#001b61;">- 0.10 Workload Risk</b>
                </div>
            </div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        st.markdown(f"<div class='panel'>{section_header('CATEM DASHBOARD <span style=\"font-size:11px;\">(Streamlit)</span>')}", unsafe_allow_html=True)
        nav_col, main_col = st.columns([0.19, 0.81], gap="small")
        with nav_col:
            st.markdown(nav_panel(), unsafe_allow_html=True)
        with main_col:
            metrics = [
                ("CATEM Score<br>(Weighted)", score_value(df, "catem_score"), "/100", "#1d6ee8"),
                ("Embodiment Score", score_value(df, "embodiment_score"), "/100", "#1d6ee8"),
                ("Presence Score", score_value(df, "presence_score"), "/100", "#1d6ee8"),
                ("Behavior Score", score_value(df, "behavior_score"), "/100", "#1d6ee8"),
                ("Workload Risk<br>(NASA-TLX)", score_value(df, "workload_risk_score"), "/100", "#ff6b1a"),
                ("System Score", score_value(df, "system_stability_score"), "/100", "#2b8c9f"),
            ]
            st.markdown("<div class='metric-grid'>" + "".join(metric_card(*item) for item in metrics) + "</div>", unsafe_allow_html=True)

            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(top_charts(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(scatter_charts(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(telemetry_chart(df), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f"<div class='panel'>{section_header('VALIDATION & ANALYSIS')}<div class='panel-body'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-card">
                <h4>Objective</h4>
                Test whether CATEM better explains/predicts telepresence quality compared to single metrics.
                <hr>
                <h4>Validation Methods</h4>
                <ul>
                    <li>Literature-Based Weighting</li>
                    <li>Correlation Analysis</li>
                    <li>Multiple Linear Regression</li>
                    <li>Random Forest Regression</li>
                    <li>Feature Importance</li>
                    <li>Prediction Accuracy (R2, MAE, RMSE)</li>
                </ul>
            </div>
            <div class="info-card"><h4>Comparison (Example Result)</h4>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(comparison_chart(), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            """
            </div>
            <div class="info-card"><h4>Correlation Matrix</h4>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(correlation_chart(df), use_container_width=True, config={"displayModeBar": False})
        relation_rows = selected_research_correlations(df)
        relations = "".join(
            f"<li>{row.relationship}: <b>{row.correlation:.2f}</b></li>"
            for row in relation_rows.itertuples(index=False)
        )
        st.markdown(
            f"""
            <ul>{relations}</ul>
            </div>
            <div class="info-card"><h4>Top Predictors of Telepresence Quality</h4>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(feature_importance_chart(df), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""
            </div>
            {explainability_card(df)}
            </div></div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
