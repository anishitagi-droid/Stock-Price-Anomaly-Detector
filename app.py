"""
app.py
Streamlit dashboard for the Stock Anomaly Detector.
 
Run with:
    streamlit run app.py
"""
 
import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).parent / "src"))
 
import streamlit as st
import pandas as pd
from datetime import date, timedelta
 
from fetch_data import fetch_and_save
from preprocess import preprocess, FEATURE_COLS, PROCESSED_DIR
from detect_anomalies import detect, summary, FLAG_COLS, SCORE_COLS
from visualize import (
    candlestick_anomalies,
    ensemble_score_plot,
    detector_heatmap,
    return_distribution,
    feature_boxplots,
    volatility_chart,
)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.title("📈 Stock Anomaly Detector")
st.caption("Isolation Forest · LOF · DBSCAN · Z-score/IQR ensemble")
 
# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
 
    ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
 
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=date(2018, 1, 1))
    end_date   = col2.date_input("End date",   value=date.today())
 
    st.divider()
    st.subheader("Detector parameters")
    contamination  = st.slider("Contamination (IF & LOF)", 0.01, 0.20, 0.05, 0.01,
                                help="Expected fraction of anomalies in the dataset.")
    vote_threshold = st.slider("Ensemble vote threshold", 1, 4, 2,
                                help="Min detectors that must agree to flag a point.")
 
    st.divider()
    run_btn = st.button("🚀 Run Detection", use_container_width=True, type="primary")
 
# ── State ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}   # keyed by ticker
 
# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.sidebar.error("Please enter a ticker symbol.")
        st.stop()
 
    with st.spinner(f"Fetching {ticker} data…"):
        try:
            fetch_and_save(ticker, start=str(start_date), end=str(end_date))
        except ValueError as e:
            st.error(str(e)); st.stop()
 
    with st.spinner("Preprocessing…"):
        preprocess(ticker)
 
    with st.spinner("Running anomaly detectors…"):
        df = detect(ticker, contamination=contamination, vote_threshold=vote_threshold)
 
    st.session_state.results[ticker] = df
    st.success(f"Done! {int(df['ensemble_anomaly'].sum())} anomalies found in {len(df)} trading days.")
 
# ── Display ───────────────────────────────────────────────────────────────────
if ticker in st.session_state.results:
    df = st.session_state.results[ticker]
    anom_df = summary(df)
 
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trading days",   f"{len(df):,}")
    k2.metric("Anomalies",      f"{int(df['ensemble_anomaly'].sum()):,}")
    k3.metric("Anomaly rate",   f"{df['ensemble_anomaly'].mean()*100:.1f}%")
    k4.metric("Avg score (anomalies)", f"{anom_df['ensemble_score'].mean():.3f}" if len(anom_df) else "—")
 
    st.divider()
 
    # ── Chart tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🕯️ Candlestick", "📊 Score", "🔥 Detector Heatmap", "📉 Returns", "📦 Features"]
    )
 
    with tab1:
        st.plotly_chart(candlestick_anomalies(df, ticker), use_container_width=True)
        st.plotly_chart(volatility_chart(df, ticker), use_container_width=True)
 
    with tab2:
        st.plotly_chart(ensemble_score_plot(df, ticker), use_container_width=True)
 
        st.subheader("Detector vote breakdown")
        vote_counts = df["vote_count"].value_counts().sort_index()
        st.bar_chart(vote_counts)
 
    with tab3:
        st.plotly_chart(detector_heatmap(df, ticker), use_container_width=True)
 
        st.subheader("Per-detector anomaly counts")
        det_summary = pd.Series(
            {c.replace("_anomaly", "").upper(): int(df[c].sum()) for c in FLAG_COLS if c in df.columns}
        ).rename("Anomalies flagged")
        st.dataframe(det_summary.to_frame(), use_container_width=True)
 
    with tab4:
        st.plotly_chart(return_distribution(df, ticker), use_container_width=True)
 
    with tab5:
        st.plotly_chart(feature_boxplots(df, ticker=ticker), use_container_width=True)
 
    st.divider()
 
    # ── Anomaly table ─────────────────────────────────────────────────────────
    st.subheader(f"🚨 Anomaly Log — {ticker}")
    st.dataframe(
        anom_df.style.background_gradient(subset=["ensemble_score"], cmap="Reds"),
        use_container_width=True,
        height=380,
    )
 
    # ── Download ──────────────────────────────────────────────────────────────
    csv = df.to_csv().encode()
    st.download_button(
        "⬇️ Download full results CSV",
        data=csv,
        file_name=f"{ticker}_anomalies.csv",
        mime="text/csv",
    )
 
else:
    st.info("👈 Configure settings in the sidebar and click **Run Detection** to get started.")
 
    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Four detectors run in parallel on engineered features:**
 
| Detector | Method | Strengths |
|---|---|---|
| **Isolation Forest** | Tree-based random partitioning | Fast, handles high dimensions |
| **LOF** | Local density comparison | Catches local outliers |
| **DBSCAN** | Density-based clustering | No assumed distribution |
| **Z-score / IQR** | Statistical thresholds | Interpretable baseline |
 
**Ensemble:** Each detector casts a vote. Points receiving ≥ threshold votes  
are flagged. An ensemble score (0–1) ranks severity.
 
**Features used:** daily return, 5-day return, log return, 5/20-day volatility,  
volume ratio, Bollinger z-score, intraday range.
        """)
 
