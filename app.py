"""Streamlit web application for Stock Price Anomaly Detection."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fetch_data import get_stock_data
from src.preprocess import preprocess_data
from src.detect_anomalies import detect_anomalies
from src.visualize import plot_interactive_price_anomalies, plot_returns_distribution, plot_volatility

# Page configuration
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>📈 Stock Price Anomaly Detector</h1>", unsafe_allow_html=True)
st.markdown("Detect unusual price movements and anomalies in stock data using machine learning.")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker input
    ticker = st.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, GOOGL, MSFT"
    ).upper()
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    
    st.markdown("---")
    st.header("🔍 Detection Methods")
    
    # Select anomaly detection methods
    methods = st.multiselect(
        "Select Detection Methods",
        ["zscore", "moving_average", "isolation_forest", "lof"],
        default=["zscore", "isolation_forest", "lof"]
    )
    
    # Detection parameters
    if "zscore" in methods:
        zscore_threshold = st.slider(
            "Z-Score Threshold",
            1.0, 5.0, 3.0, 0.1
        )
    
    if "moving_average" in methods:
        ma_window = st.slider(
            "MA Window Size",
            5, 50, 20, 5
        )
        ma_threshold = st.slider(
            "MA Threshold (std)",
            1.0, 5.0, 2.0, 0.1
        )
    
    if "isolation_forest" in methods:
        if_contamination = st.slider(
            "IF Contamination",
            0.01, 0.5, 0.05, 0.01
        )
    
    if "lof" in methods:
        lof_contamination = st.slider(
            "LOF Contamination",
            0.01, 0.5, 0.05, 0.01
        )

# Main content
try:
    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        raw_data = get_stock_data(
            ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        processed_data = preprocess_data(raw_data)
    
    # Build kwargs for detection
    kwargs = {}
    if "zscore" in methods:
        kwargs['threshold'] = zscore_threshold
    if "moving_average" in methods:
        kwargs['window'] = ma_window
        kwargs['ma_threshold'] = ma_threshold
    if "isolation_forest" in methods:
        kwargs['contamination'] = if_contamination
    if "lof" in methods:
        kwargs['contamination'] = lof_contamination
    
    # Detect anomalies
    with st.spinner("Detecting anomalies..."):
        result = detect_anomalies(processed_data, methods=methods, **kwargs)
    
    # Display results
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", len(result))
    with col2:
        st.metric("Anomalies Detected", result['is_anomaly'].sum())
    with col3:
        st.metric("Anomaly %", f"{(result['is_anomaly'].sum() / len(result) * 100):.2f}%")
    with col4:
        st.metric("Latest Price", f"${result['Close'].iloc[-1]:.2f}")
    
    st.markdown("---")
    
    # Interactive plot
    st.subheader("Price Chart with Anomalies")
    fig = plot_interactive_price_anomalies(result, title=f"{ticker} Stock Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Anomaly Details", "Returns Analysis", "Volatility"])
    
    with tab1:
        st.subheader("Detected Anomalies")
        anomalies_df = result[result['is_anomaly']][[
            'Date', 'Close', 'Daily_Return', 'Volatility'
        ]].head(20)
        st.dataframe(anomalies_df, use_container_width=True)
        
        # Download results
        csv = result.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv,
            file_name=f"{ticker}_anomalies.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Returns Distribution")
        if 'Daily_Return' in result.columns:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(result['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Daily Return')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Daily Returns')
            axes[0].grid(True, alpha=0.3)
            
            from scipy import stats
            stats.probplot(result['Daily_Return'].dropna(), dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot')
            axes[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Volatility Analysis")
        if 'Volatility' in result.columns:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(result['Date'], result['Volatility'], color='orange', linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.set_title('20-Day Rolling Volatility')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Built with Streamlit | Data from Yahoo Finance | © 2024
    </div>
""", unsafe_allow_html=True)
