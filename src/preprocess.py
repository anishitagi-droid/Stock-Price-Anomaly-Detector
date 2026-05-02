"""Module for preprocessing and cleaning stock price data."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def preprocess_data(
    data: pd.DataFrame,
    handle_missing: str = 'forward_fill',
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Preprocess stock price data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw stock price data
    handle_missing : str
        Method to handle missing values: 'forward_fill', 'interpolate', 'drop'
    remove_duplicates : bool
        Whether to remove duplicate entries
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed data
    """
    df = data.copy()
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates(subset=['Date'], keep='first')
    
    # Handle missing values
    if handle_missing == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif handle_missing == 'interpolate':
        df = df.interpolate(method='linear')
    elif handle_missing == 'drop':
        df = df.dropna()
    
    # Ensure numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate additional features
    df = calculate_features(df)
    
    return df


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical features from price data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed stock data
    
    Returns:
    --------
    pd.DataFrame
        Data with additional features
    """
    df = data.copy()
    
    if 'Close' in df.columns:
        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Price range
        if 'High' in df.columns and 'Low' in df.columns:
            df['Price_Range'] = df['High'] - df['Low']
            df['Price_Range_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    return df


def normalize_features(
    data: pd.DataFrame,
    method: str = 'minmax',
    features: Optional[list] = None
) -> pd.DataFrame:
    """
    Normalize specified features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to normalize
    method : str
        Normalization method: 'minmax' or 'zscore'
    features : list, optional
        Features to normalize. If None, normalizes numeric columns.
    
    Returns:
    --------
    pd.DataFrame
        Data with normalized features
    """
    df = data.copy()
    
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'minmax':
        for col in features:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val - min_val != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        for col in features:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df[col] = (df[col] - mean_val) / std_val
    
    return df


if __name__ == "__main__":
    from fetch_data import get_stock_data
    
    # Example usage
    raw_data = get_stock_data('AAPL')
    processed_data = preprocess_data(raw_data)
    print(processed_data.head())
    print(f"\nFeatures: {processed_data.columns.tolist()}")
