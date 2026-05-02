"""
visualize.py
Reusable Plotly chart functions consumed by both the notebook and app.py.
All functions return a plotly Figure so callers can .show() or st.plotly_chart().
"""
 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
 
 
# ── Colour palette ────────────────────────────────────────────────────────────
NORMAL_CLR  = "rgba(99, 149, 237, 0.55)"
ANOMALY_CLR = "#ef4444"
MA_CLR      = "#f59e0b"
GRID_CLR    = "rgba(255,255,255,0.07)"
 
 
# ── 1 · Candlestick + anomaly markers ────────────────────────────────────────
 
def candlestick_anomalies(
    df: pd.DataFrame,
    ticker: str = "",
    anomaly_col: str = "ensemble_anomaly",
    score_col: str | None = "ensemble_score",
) -> go.Figure:
    """
    Two-row figure: OHLC candles (+ anomaly markers) over volume bars.
 
    Args:
        df:           Processed + detected DataFrame.
        ticker:       Symbol string used in title.
        anomaly_col:  Boolean column that flags anomalies.
        score_col:    Optional float column added to hover text.
    """
    anom = df[df[anomaly_col]]
 
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.03,
        subplot_titles=("", "Volume"),
    )
 
    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC", increasing_line_color="#22c55e",
            decreasing_line_color=ANOMALY_CLR,
        ),
        row=1, col=1,
    )
 
    # Anomaly scatter
    hover = (
        anom[score_col].round(3).astype(str)
        if score_col and score_col in anom.columns
        else None
    )
    fig.add_trace(
        go.Scatter(
            x=anom.index, y=anom["High"] * 1.015,
            mode="markers",
            marker=dict(symbol="triangle-down", color=ANOMALY_CLR, size=10),
            name="Anomaly",
            hovertext=hover,
            hovertemplate="<b>%{x}</b><br>Score: %{hovertext}<extra></extra>",
        ),
        row=1, col=1,
    )
 
    # Volume bars
    vol_colors = np.where(df[anomaly_col], ANOMALY_CLR, NORMAL_CLR)
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"),
        row=2, col=1,
    )
 
    fig.update_layout(
        title=f"{ticker} – Anomaly Detection",
        xaxis_rangeslider_visible=False,
        height=580,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR)
    return fig
 
 
# ── 2 · Ensemble score time-series ───────────────────────────────────────────
 
def ensemble_score_plot(
    df: pd.DataFrame,
    ticker: str = "",
    score_col: str = "ensemble_score",
    anomaly_col: str = "ensemble_anomaly",
    threshold: float | None = None,
) -> go.Figure:
    """Line chart of ensemble anomaly score with threshold band."""
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=df.index, y=df[score_col],
        mode="lines", name="Ensemble score",
        line=dict(color="#818cf8", width=1.5),
    ))
 
    anom = df[df[anomaly_col]]
    fig.add_trace(go.Scatter(
        x=anom.index, y=anom[score_col],
        mode="markers", name="Anomaly",
        marker=dict(color=ANOMALY_CLR, size=7),
    ))
 
    if threshold is not None:
        fig.add_hline(
            y=threshold, line_dash="dot",
            line_color=MA_CLR, annotation_text=f"threshold={threshold:.2f}",
        )
 
    fig.update_layout(
        title=f"{ticker} – Ensemble Anomaly Score",
        yaxis_title="Score (0–1)",
        template="plotly_dark",
        height=340,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig
 
 
# ── 3 · Detector comparison heatmap ──────────────────────────────────────────
 
def detector_heatmap(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """
    Binary heatmap: rows = detectors, columns = dates.
    Only shows the portion of the timeline where at least one detector fires.
    """
    flag_cols = [c for c in ["if_anomaly", "lof_anomaly", "dbscan_anomaly", "baseline_anomaly"] if c in df.columns]
    labels    = [c.replace("_anomaly", "").upper() for c in flag_cols]
 
    active = df[df[flag_cols].any(axis=1)]
    z      = active[flag_cols].astype(int).T.values
 
    fig = go.Figure(go.Heatmap(
        z=z, x=active.index, y=labels,
        colorscale=[[0, "rgba(30,30,50,0.6)"], [1, ANOMALY_CLR]],
        showscale=False,
        hovertemplate="Date: %{x}<br>Detector: %{y}<extra></extra>",
    ))
 
    fig.update_layout(
        title=f"{ticker} – Detector Agreement (anomaly days only)",
        template="plotly_dark",
        height=260,
        margin=dict(l=80, r=30, t=50, b=40),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig
 
 
# ── 4 · Return distribution + anomaly rug ────────────────────────────────────
 
def return_distribution(
    df: pd.DataFrame,
    ticker: str = "",
    return_col: str = "return_1d",
    anomaly_col: str = "ensemble_anomaly",
) -> go.Figure:
    """Histogram of daily returns with anomalies shown as a rug plot."""
    normal = df.loc[~df[anomaly_col], return_col].dropna()
    anom   = df.loc[df[anomaly_col],  return_col].dropna()
 
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=normal, nbinsx=120, name="Normal",
        marker_color=NORMAL_CLR, opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=anom, nbinsx=40, name="Anomaly",
        marker_color=ANOMALY_CLR, opacity=0.9,
    ))
    # Rug
    fig.add_trace(go.Scatter(
        x=anom, y=[-5] * len(anom),
        mode="markers", marker=dict(symbol="line-ns-open", color=ANOMALY_CLR, size=10),
        name="Anomaly rug",
    ))
 
    fig.update_layout(
        barmode="overlay",
        title=f"{ticker} – 1-Day Return Distribution",
        xaxis_title="Return", yaxis_title="Count",
        template="plotly_dark",
        height=360,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig
 
 
# ── 5 · Feature box-plots: anomaly vs normal ─────────────────────────────────
 
def feature_boxplots(
    df: pd.DataFrame,
    features: list[str] | None = None,
    anomaly_col: str = "ensemble_anomaly",
    ticker: str = "",
) -> go.Figure:
    """Side-by-side box plots comparing anomalous vs normal days per feature."""
    from preprocess import FEATURE_COLS
    features = features or FEATURE_COLS
 
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=features,
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )
 
    for i, feat in enumerate(features[:8]):
        row, col = divmod(i, 4)
        for flag, name, color in [(False, "Normal", NORMAL_CLR), (True, "Anomaly", ANOMALY_CLR)]:
            subset = df.loc[df[anomaly_col] == flag, feat].dropna()
            fig.add_trace(
                go.Box(y=subset, name=name, marker_color=color,
                       showlegend=(i == 0), legendgroup=name),
                row=row + 1, col=col + 1,
            )
 
    fig.update_layout(
        title=f"{ticker} – Feature Distribution: Normal vs Anomaly",
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=20, t=80, b=40),
        boxmode="group",
    )
    return fig
 
 
# ── 6 · Rolling volatility + anomaly shading ─────────────────────────────────
 
def volatility_chart(
    df: pd.DataFrame,
    ticker: str = "",
    anomaly_col: str = "ensemble_anomaly",
) -> go.Figure:
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=df.index, y=df["volatility_20d"],
        fill="tozeroy", mode="lines",
        line=dict(color="#818cf8", width=1.5),
        fillcolor="rgba(129,140,248,0.2)", name="20d Vol",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["volatility_5d"],
        mode="lines", line=dict(color=MA_CLR, width=1),
        name="5d Vol",
    ))
 
    # Shade anomaly regions
    for date in df.index[df[anomaly_col]]:
        fig.add_vrect(
            x0=date, x1=date,
            fillcolor=ANOMALY_CLR, opacity=0.25, line_width=0,
        )
 
    fig.update_layout(
        title=f"{ticker} – Rolling Volatility",
        yaxis_title="Std Dev (log return)",
        template="plotly_dark",
        height=340,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig
 
