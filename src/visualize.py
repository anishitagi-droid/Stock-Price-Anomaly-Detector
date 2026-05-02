"""Module for visualizing stock price data and anomalies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def plot_price_and_anomalies(
    data: pd.DataFrame,
    title: str = 'Stock Price with Detected Anomalies',
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot stock price with anomalies highlighted.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with anomaly detection results
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    if 'Date' in data.columns:
        x = data['Date']
    else:
        x = range(len(data))
    
    ax.plot(x, data['Close'], label='Close Price', linewidth=2, color='blue')
    
    # Highlight anomalies
    if 'is_anomaly' in data.columns:
        anomalies = data[data['is_anomaly']]
        ax.scatter(
            anomalies.index if 'Date' not in data.columns else anomalies['Date'],
            anomalies['Close'],
            color='red',
            s=100,
            label='Anomalies',
            zorder=5
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_interactive_price_anomalies(
    data: pd.DataFrame,
    title: str = 'Stock Price Anomaly Detection'
) -> go.Figure:
    """
    Create interactive plot using Plotly.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with anomaly detection results
    title : str
        Plot title
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data['Date'] if 'Date' in data.columns else data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add anomalies
    if 'is_anomaly' in data.columns:
        anomalies = data[data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies['Date'] if 'Date' in anomalies.columns else anomalies.index,
            y=anomalies['Close'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=600
    )
    
    return fig


def plot_returns_distribution(
    data: pd.DataFrame,
    figsize: tuple = (12, 4)
) -> None:
    """
    Plot distribution of daily returns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Daily_Return column
    figsize : tuple
        Figure size
    """
    if 'Daily_Return' not in data.columns:
        print("Daily_Return column not found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(data['Daily_Return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Daily Return', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(data['Daily_Return'].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot of Daily Returns', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_anomaly_methods_comparison(
    data: pd.DataFrame,
    figsize: tuple = (14, 6)
) -> None:
    """
    Compare different anomaly detection methods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with multiple anomaly detection columns
    figsize : tuple
        Figure size
    """
    anomaly_cols = [col for col in data.columns if col.startswith('anomaly_')]
    
    if not anomaly_cols:
        print("No anomaly detection columns found")
        return
    
    fig, axes = plt.subplots(len(anomaly_cols) + 1, 1, figsize=(figsize[0], figsize[1] * len(anomaly_cols)))
    
    if 'Date' in data.columns:
        x = data['Date']
    else:
        x = range(len(data))
    
    # Plot price with each method
    for idx, col in enumerate(anomaly_cols):
        ax = axes[idx]
        ax.plot(x, data['Close'], label='Close Price', color='blue', linewidth=1)
        
        anomalies = data[data[col]]
        ax.scatter(
            anomalies.index if 'Date' not in data.columns else anomalies['Date'],
            anomalies['Close'],
            color='red',
            s=50,
            label='Anomalies'
        )
        ax.set_title(f'{col.replace("anomaly_", "").upper()} Detection', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot combined result
    if 'is_anomaly' in data.columns:
        ax = axes[-1]
        ax.plot(x, data['Close'], label='Close Price', color='blue', linewidth=1)
        anomalies = data[data['is_anomaly']]
        ax.scatter(
            anomalies.index if 'Date' not in data.columns else anomalies['Date'],
            anomalies['Close'],
            color='red',
            s=50,
            label='Combined Anomalies'
        )
        ax.set_title('Combined Detection', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_volatility(
    data: pd.DataFrame,
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot stock price and volatility.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock data with Volatility column
    figsize : tuple
        Figure size
    """
    if 'Volatility' not in data.columns:
        print("Volatility column not found")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    if 'Date' in data.columns:
        x = data['Date']
    else:
        x = range(len(data))
    
    # Price
    axes[0].plot(x, data['Close'], color='blue', linewidth=2, label='Close Price')
    axes[0].set_ylabel('Price ($)', fontsize=11)
    axes[0].set_title('Stock Price', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Volatility
    axes[1].plot(x, data['Volatility'], color='orange', linewidth=2, label='Volatility')
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Volatility', fontsize=11)
    axes[1].set_title('20-Day Rolling Volatility', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from fetch_data import get_stock_data
    from preprocess import preprocess_data
    from detect_anomalies import detect_anomalies
    
    # Example usage
    raw_data = get_stock_data('AAPL')
    processed_data = preprocess_data(raw_data)
    result = detect_anomalies(processed_data)
    
    plot_price_and_anomalies(result)
    plot_returns_distribution(result)
    plot_volatility(result)
