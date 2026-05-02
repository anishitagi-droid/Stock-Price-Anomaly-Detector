"""Module for detecting anomalies in stock price data."""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


def zscore_anomaly_detection(
    data: pd.DataFrame,
    column: str = 'Close',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect anomalies using Z-score method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    column : str
        Column to analyze for anomalies
    threshold : float
        Z-score threshold (default: 3.0)
    
    Returns:
    --------
    pd.DataFrame
        Data with 'anomaly' and 'zscore' columns
    """
    df = data.copy()
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean = df[column].mean()
    std = df[column].std()
    
    df['zscore'] = np.abs((df[column] - mean) / std)
    df['anomaly_zscore'] = df['zscore'] > threshold
    
    return df


def moving_average_anomaly_detection(
    data: pd.DataFrame,
    column: str = 'Close',
    window: int = 20,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies based on deviation from moving average.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    column : str
        Column to analyze
    window : int
        Moving average window size
    threshold : float
        Standard deviation threshold
    
    Returns:
    --------
    pd.DataFrame
        Data with anomaly flags
    """
    df = data.copy()
    
    ma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    
    deviation = np.abs(df[column] - ma)
    df['anomaly_ma'] = deviation > (threshold * std)
    
    return df


def isolation_forest_detection(
    data: pd.DataFrame,
    features: Optional[list] = None,
    contamination: float = 0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    features : list, optional
        Features to use for detection
    contamination : float
        Expected proportion of anomalies (0.0 to 0.5)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Data with anomaly flags
    """
    df = data.copy()
    
    if features is None:
        features = ['Daily_Return', 'Volatility'] if 'Daily_Return' in df.columns else ['Close']
    
    # Select available features
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError(f"No available features from {features}")
    
    # Prepare data
    X = df[available_features].dropna().values
    indices = df[available_features].dropna().index
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Mark anomalies (-1 = anomaly, 1 = normal)
    df['anomaly_if'] = False
    df.loc[indices, 'anomaly_if'] = predictions == -1
    
    return df


def local_outlier_factor_detection(
    data: pd.DataFrame,
    features: Optional[list] = None,
    n_neighbors: int = 20,
    contamination: float = 0.05
) -> pd.DataFrame:
    """
    Detect anomalies using Local Outlier Factor (LOF).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    features : list, optional
        Features to use for detection
    n_neighbors : int
        Number of neighbors for LOF
    contamination : float
        Expected proportion of anomalies
    
    Returns:
    --------
    pd.DataFrame
        Data with anomaly flags
    """
    df = data.copy()
    
    if features is None:
        features = ['Daily_Return', 'Volatility'] if 'Daily_Return' in df.columns else ['Close']
    
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError(f"No available features from {features}")
    
    # Prepare data
    X = df[available_features].dropna().values
    indices = df[available_features].dropna().index
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    predictions = lof.fit_predict(X_scaled)
    
    # Mark anomalies
    df['anomaly_lof'] = False
    df.loc[indices, 'anomaly_lof'] = predictions == -1
    
    return df


def detect_anomalies(
    data: pd.DataFrame,
    methods: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Detect anomalies using multiple methods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    methods : list, optional
        Anomaly detection methods to use
    **kwargs
        Additional parameters for detection methods
    
    Returns:
    --------
    pd.DataFrame
        Data with anomaly detection results
    """
    if methods is None:
        methods = ['zscore', 'isolation_forest', 'lof']
    
    df = data.copy()
    
    if 'zscore' in methods:
        df = zscore_anomaly_detection(df, **kwargs)
    
    if 'moving_average' in methods:
        df = moving_average_anomaly_detection(df, **kwargs)
    
    if 'isolation_forest' in methods:
        df = isolation_forest_detection(df, **kwargs)
    
    if 'lof' in methods:
        df = local_outlier_factor_detection(df, **kwargs)
    
    # Combined anomaly flag
    anomaly_cols = [col for col in df.columns if col.startswith('anomaly_')]
    if anomaly_cols:
        df['is_anomaly'] = df[anomaly_cols].any(axis=1)
    
    return df


if __name__ == "__main__":
    from fetch_data import get_stock_data
    from preprocess import preprocess_data
    
    # Example usage
    raw_data = get_stock_data('AAPL')
    processed_data = preprocess_data(raw_data)
    result = detect_anomalies(processed_data)
    
    print(f"Total anomalies detected: {result['is_anomaly'].sum()}")
    print(result[result['is_anomaly']][['Date', 'Close', 'is_anomaly']].head())
