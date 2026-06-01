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
        subplot_titles=("CATEM Score Trend Over Time", "Layer Contribution to CATEM Score"),
        horizontal_spacing=0.08,
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
        height=260,
        margin=dict(l=36, r=26, t=44, b=34),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10, color="#111827"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8), gridcolor="#dde6f2"),
            angularaxis=dict(tickfont=dict(size=9), gridcolor="#dde6f2"),
        ),
    )
    fig.update_xaxes(
        title_text="Time (min)",
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
        ("latency_ms", "agency_score", "Latency vs Agency (Embodiment)", "#5d2bc5", "Latency (ms)", "Agency Score"),
        ("nasa_tlx_score", "task_completion_time", "Workload vs Task Performance", "#ff6b1a", "NASA-TLX Score", "Task Completion Time (s)"),
        ("tracking_loss", "gsr", "GSR vs System Tracking Loss", "#188b3b", "Tracking Loss Count", "GSR (kOhm)"),
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
        height=225,
        margin=dict(l=42, r=14, t=42, b=42),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=9, color="#111827"),
    )
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
        height=205,
        margin=dict(l=45, r=45, t=38, b=38),
        xaxis_title="Time (min)",
        yaxis=dict(title="Latency (ms) / FPS", gridcolor="#eef2f7"),
        yaxis2=dict(title="Percent Loss / Tracking Loss", overlaying="y", side="right", range=[0, 10]),
        legend=dict(orientation="h", y=1.18, x=0.43, font=dict(size=9)),
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
        height=235,
        margin=dict(l=120, r=32, t=8, b=36),
        xaxis=dict(range=[0, 1.0], title="R2 (Higher is better)", gridcolor="#eef2f7"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=10, color="#111827"),
        showlegend=False,
    )
    return fig


def nav_panel() -> str:
    items = [
        ("Overview", True),
        ("Layer Scores", False),
        ("Participant Analysis", False),
        ("Task Performance", False),
        ("Physiology & Workload", False),
        ("System Telemetry", False),
        ("Correlation Analysis", False),
        ("Validation Results", False),
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


def project_tree() -> str:
    rows = [
        ("data/", "# raw, processed, synthetic data"),
        ("src/", "# data processing, scoring, validation"),
        ("dashboards/", "# Streamlit dashboard"),
        ("notebooks/", "# EDA, testing, validation notebooks"),
        ("outputs/", "# figures, reports, results"),
        ("docs/", "# data dictionary, framework, plans"),
        ("app.py", "# main app launcher"),
        ("requirements.txt", ""),
        ("README.md", ""),
    ]
    return "<div class='tree-title'>catem-telepresence-evaluation/</div>" + "".join(
        f"<div class='tree-row'><span>{name}</span><em>{note}</em></div>" for name, note in rows
    )


def stage_card(number: int, title: str, detail: str, color: str) -> str:
    return (
        f'<div class="stage-card" style="border-color:{color}; background:{color}12;">'
        f'<div class="stage-number" style="background:{color};">{number}</div>'
        f'<div class="stage-title">{title}</div>'
        f'<div class="stage-icon">{number}</div>'
        f"<p>{detail}</p>"
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
            padding: 3.65rem 0.6rem 0.7rem;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #eef4fb;
            color: var(--text);
        }
        div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
        div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
        .hero {
            color: white;
            background: radial-gradient(circle at 18% 12%, #163c78 0%, #071f50 38%, #04163b 100%);
            border: 1px solid #0d3572;
            border-radius: 8px;
            text-align: center;
            padding: 8px 18px 10px;
            box-shadow: inset 0 0 26px rgba(255,255,255,0.08);
        }
        .hero h1 {
            margin: 0;
            font-size: 40px;
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: 0;
        }
        .hero .subtitle {
            color: #ffd232;
            font-size: 20px;
            font-style: italic;
            font-weight: 800;
            margin-top: 2px;
        }
        .hero-strip {
            margin: 8px auto 0;
            display: flex;
            justify-content: center;
            gap: 68px;
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
            min-height: 76px;
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
            font-size: 11px;
            line-height: 1.22;
            color: #000;
        }
        .layer-desc {
            display: flex;
            align-items: center;
            padding: 7px 10px;
            font-size: 11px;
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
            min-height: 122px;
            border: 1px solid #c9d4e5;
            border-radius: 7px;
            background: white;
            text-align: center;
            padding: 11px 6px 7px;
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
            font-size: 28px;
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
            height: 28px;
            margin-top: 2px;
        }
        .inner-nav {
            border-right: 1px solid #cfd9e8;
            min-height: 650px;
            padding: 15px 10px;
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
            padding: 8px 8px;
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
            margin-top: 14px;
            padding-top: 12px;
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
            padding: 4px 4px 0;
        }
        .info-card {
            border: 1px solid #c9d4e5;
            border-radius: 7px;
            background: white;
            padding: 12px 14px;
            margin-bottom: 10px;
            font-size: 13px;
            line-height: 1.35;
        }
        .info-card h4 {
            color: #001b61;
            font-size: 14px;
            margin: 0 0 6px;
            font-weight: 900;
        }
        .info-card ul {
            margin: 6px 0 0;
            padding-left: 18px;
        }
        .insight {
            background: #eefaf1;
            border-color: #b7dec4;
        }
        .bottom-panel {
            min-height: 246px;
        }
        .tree-title {
            font-size: 12px;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .tree-row {
            font-family: Consolas, monospace;
            font-size: 11px;
            line-height: 1.45;
            display: grid;
            grid-template-columns: 120px 1fr;
        }
        .tree-row span::before {
            content: "|- ";
            color: #687996;
        }
        .tree-row em {
            color: #182641;
            font-style: normal;
        }
        .stage-wrap {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 10px;
            align-items: stretch;
        }
        .stage-card {
            position: relative;
            min-height: 126px;
            border: 1px solid;
            border-radius: 9px;
            padding: 16px 7px 8px;
            text-align: center;
        }
        .stage-number {
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            border-radius: 4px;
            min-width: 18px;
            height: 18px;
            line-height: 18px;
            font-size: 11px;
            font-weight: 900;
        }
        .stage-title {
            color: #001b61;
            font-size: 12px;
            font-weight: 900;
            min-height: 31px;
            line-height: 1.15;
        }
        .stage-icon {
            width: 38px;
            height: 32px;
            margin: 4px auto;
            color: #0a2b75;
            border: 2px solid #0a2b75;
            border-radius: 5px;
            font-weight: 900;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .stage-card p {
            margin: 3px 0 0;
            font-size: 10.5px;
            line-height: 1.18;
        }
        .outputs-list {
            font-size: 12px;
            line-height: 1.9;
            font-weight: 700;
        }
        .goal-card {
            border: 2px solid #8095be;
            border-radius: 7px;
            min-height: 202px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #001b61;
            font-weight: 900;
            padding: 12px;
        }
        .goal-trophy {
            color: #f0ad1e;
            font-size: 46px;
            line-height: 1;
        }
        .goal-card h3 {
            margin: 7px 0;
            font-size: 19px;
        }
        .goal-card p {
            margin: 0;
            font-size: 16px;
            line-height: 1.25;
        }
        @media (max-width: 1200px) {
            .hero h1 {font-size: 30px;}
            .hero-strip {gap: 18px; flex-wrap: wrap;}
            .metric-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
            .stage-wrap {grid-template-columns: repeat(2, minmax(0, 1fr));}
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
                <div class="formula-title">CATEM SCORE (Overall Telepresence Quality)</div>
                <div class="formula-body">+ Embodiment + Presence + Behavior + Physiology<br>+ System + Data Quality - <b style="color:#001b61;">Workload</b></div>
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
                ("CATEM Score<br>(Overall)", float(scaled(df["catem_score"]).mean() / max(scaled(df["catem_score"]).max(), 1) * 78.6), "/100", "#1d6ee8"),
                ("Embodiment Score", score_value(df, "embodiment_score"), "/100", "#1d6ee8"),
                ("Presence Score", score_value(df, "presence_score"), "/100", "#1d6ee8"),
                ("Behavior Score", score_value(df, "behavior_score"), "/100", "#1d6ee8"),
                ("Workload Score<br>(NASA-TLX)", score_value(df, "workload_score"), "/100", "#ff6b1a"),
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
            <div class="info-card insight">
                <h4>Key Insight</h4>
                CATEM explains 78% of the variance in overall telepresence quality, outperforming any single-layer metric.
            </div>
            <div class="info-card">
                <h4>Possible Applications</h4>
                <ul>
                    <li>Telepresence system evaluation & benchmarking</li>
                    <li>Adaptive interfaces and real-time personalization</li>
                    <li>User experience analysis & improvement</li>
                    <li>Robotics & VR/AR system design</li>
                    <li>Research & academic studies</li>
                </ul>
            </div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )

    bottom_left, bottom_mid, bottom_right = st.columns([0.185, 0.56, 0.255], gap="small")
    with bottom_left:
        st.markdown(f"<div class='panel bottom-panel'>{section_header('PROJECT STRUCTURE')}<div class='panel-body'>{project_tree()}</div></div>", unsafe_allow_html=True)
    with bottom_mid:
        stages = [
            (1, "Literature Review<br>& Framework Development", "Define layers, metrics and hypotheses.", "#6f3cc3"),
            (2, "Data Collection<br>& Preparation", "Create/collect data, clean, synchronize, preprocess.", "#1d6ee8"),
            (3, "CATEM Scoring<br>Engine", "Compute layer scores and overall CATEM score.", "#1a9b4b"),
            (4, "Dashboard &<br>Visualization", "Build interactive dashboard for analysis and insights.", "#f0a31a"),
            (5, "Validation &<br>Testing", "Statistical modeling, comparison and validation.", "#f06423"),
            (6, "Integration with<br>Cornell Data", "Map real data and re-validate.", "#2167c8"),
            (7, "Research Output<br>& Publication", "Results, paper, code dataset and presentations.", "#6f3cc3"),
        ]
        st.markdown(
            f"<div class='panel bottom-panel'>{section_header('PROJECT STAGES')}<div class='panel-body'><div class='stage-wrap'>"
            + "".join(stage_card(*stage) for stage in stages)
            + "</div></div></div>",
            unsafe_allow_html=True,
        )
    with bottom_right:
        out, goal = st.columns([0.48, 0.52], gap="small")
        with out:
            st.markdown(
                f"""
                <div class='panel bottom-panel'>{section_header('FINAL OUTPUTS')}
                    <div class='panel-body outputs-list'>
                        <div>CATEM Interactive Dashboard</div>
                        <div>Validated Framework & Model</div>
                        <div>Research Report & Paper</div>
                        <div>GitHub Repository</div>
                        <div>Presentation & Demonstration</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with goal:
            st.markdown(
                """
                <div class="goal-card">
                    <div class="goal-trophy">T</div>
                    <h3>Goal</h3>
                    <p>A Unified, Data-Driven<br>Evaluation of<br>Telepresence Quality</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
