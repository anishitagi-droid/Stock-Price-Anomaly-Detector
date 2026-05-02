"""Module for fetching stock price data from various sources."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List


def get_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch historical stock price data using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'. Defaults to 1 year ago.
    end_date : str, optional
        End date in format 'YYYY-MM-DD'. Defaults to today.
    interval : str, optional
        Data interval ('1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
        Default is '1d' for daily data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch data
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # Reset index to make Date a column
        data.reset_index(inplace=True)
        
        return data
    
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def get_multiple_stocks(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """
    Fetch data for multiple stocks.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock ticker symbols
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    dict
        Dictionary with ticker as key and DataFrame as value
    """
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = get_stock_data(ticker, start_date, end_date)
            print(f"✓ Fetched data for {ticker}")
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")
    
    return data


if __name__ == "__main__":
    # Example usage
    df = get_stock_data('AAPL', start_date='2023-01-01', end_date='2024-01-01')
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
