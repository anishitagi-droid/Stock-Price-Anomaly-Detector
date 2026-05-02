"""
fetch_data.py
Downloads OHLCV stock data using yfinance and saves raw CSVs to data/raw/.
"""
 
import yfinance as yf
import pandas as pd
from pathlib import Path
 
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
 
 
import yfinance as yf
import time

def fetch_stock(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    time.sleep(2)  # avoid rate limit
    
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)
    
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    
    df.index = df.index.tz_localize(None)  # remove timezone
    df.index.name = "Date"
    return df
 
 
def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    """Save DataFrame to data/raw/<TICKER>.csv."""
    path = RAW_DIR / f"{ticker.upper()}.csv"
    df.to_csv(path)
    print(f"[fetch_data] Saved {len(df)} rows → {path}")
    return path
 
 
def fetch_and_save(ticker: str, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    """Convenience wrapper: fetch + save, return DataFrame."""
    df = fetch_stock(ticker, start, end)
    save_raw(df, ticker)
    return df
 
 
if __name__ == "__main__":
    # Quick smoke-test: fetch a handful of well-known tickers
    tickers = ["AAPL", "TSLA", "SPY"]
    for t in tickers:
        fetch_and_save(t, start="2018-01-01")
 
