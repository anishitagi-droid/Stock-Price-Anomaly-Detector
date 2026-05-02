"""
preprocess.py
Loads raw OHLCV CSVs, engineers features used by anomaly detectors,
and saves processed CSVs to data/processed/.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
 
# ── Feature engineering ──────────────────────────────────────────────────────
 
def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1d"]  = df["Close"].pct_change()
    df["return_5d"]  = df["Close"].pct_change(5)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df
 
 
def add_volatility(df: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"volatility_{w}d"] = df["log_return"].rolling(w).std()
    return df
 
 
def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["volume_ma"]    = df["Volume"].rolling(window).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"]   # spike indicator
    return df
 
 
def add_price_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["ma"]       = df["Close"].rolling(window).mean()
    df["std"]      = df["Close"].rolling(window).std()
    df["z_score"]  = (df["Close"] - df["ma"]) / df["std"]   # Bollinger z-score
    df["range"]    = (df["High"] - df["Low"]) / df["Close"]  # intraday range %
    return df
 
 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = df.copy()
    df = add_returns(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df.dropna(inplace=True)
    return df
 
 
# ── I/O helpers ───────────────────────────────────────────────────────────────
 
def load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}. Run fetch_data.py first.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    # yfinance MultiIndex columns come out as tuples – flatten if needed
    if isinstance(df.columns[0], tuple):
        df.columns = [c[0] for c in df.columns]
    return df
 
 
def save_processed(df: pd.DataFrame, ticker: str) -> Path:
    path = PROCESSED_DIR / f"{ticker.upper()}_processed.csv"
    df.to_csv(path)
    print(f"[preprocess] Saved {len(df)} rows, {len(df.columns)} cols → {path}")
    return path
 
 
def preprocess(ticker: str) -> pd.DataFrame:
    """Load raw → engineer features → save processed → return DataFrame."""
    df = load_raw(ticker)
    df = engineer_features(df)
    save_processed(df, ticker)
    return df
 
 
# ── Feature list used by detectors ───────────────────────────────────────────
FEATURE_COLS = [
    "return_1d", "return_5d", "log_return",
    "volatility_5d", "volatility_20d",
    "volume_ratio",
    "z_score", "range",
]
 
 
if __name__ == "__main__":
    for ticker in ["AAPL", "TSLA", "SPY"]:
        try:
            preprocess(ticker)
        except FileNotFoundError as e:
            print(e)
 
