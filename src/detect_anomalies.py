"""
detect_anomalies.py
Runs multiple anomaly detection algorithms on preprocessed features
and produces an ensemble anomaly score saved to data/processed/.
 
Algorithms:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - DBSCAN
  - Z-score / IQR baseline (from preprocess)
  - Ensemble: majority vote + mean score
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
 
# Re-use the feature list defined in preprocess
import sys
sys.path.append(str(Path(__file__).parent))
from preprocess import FEATURE_COLS, PROCESSED_DIR, preprocess, load_raw
 
RESULTS_DIR = PROCESSED_DIR  # save alongside other processed files
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _scale(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return StandardScaler-transformed feature matrix."""
    return StandardScaler().fit_transform(df[cols])
 
 
# ── Individual detectors ──────────────────────────────────────────────────────
 
def isolation_forest(
    df: pd.DataFrame,
    cols: list[str] = FEATURE_COLS,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """
    Returns a Series of anomaly scores (higher = more anomalous).
    IsolationForest raw scores are negated so that high = anomalous.
    Binary flag stored as 'if_anomaly'.
    """
    X = _scale(df, cols)
    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(X)
    # decision_function: lower (more negative) = more anomalous → negate
    scores = -clf.decision_function(X)
    labels = clf.predict(X)           # -1 = anomaly, 1 = normal
    return pd.DataFrame(
        {"if_score": scores, "if_anomaly": labels == -1},
        index=df.index,
    )
 
 
def local_outlier_factor(
    df: pd.DataFrame,
    cols: list[str] = FEATURE_COLS,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    LOF: higher score = more anomalous.
    Note: LOF is transductive — no predict on new data.
    """
    X = _scale(df, cols)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
    labels = clf.fit_predict(X)       # -1 = anomaly
    scores = -clf.negative_outlier_factor_   # negate so high = anomalous
    return pd.DataFrame(
        {"lof_score": scores, "lof_anomaly": labels == -1},
        index=df.index,
    )
 
 
def dbscan_detector(
    df: pd.DataFrame,
    cols: list[str] = FEATURE_COLS,
    eps: float = 1.5,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    DBSCAN labels noise points (-1) as anomalies.
    Score = 1 for anomaly, 0 for cluster member.
    """
    X = _scale(df, cols)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    anomaly = labels == -1
    return pd.DataFrame(
        {"dbscan_score": anomaly.astype(float), "dbscan_anomaly": anomaly},
        index=df.index,
    )
 
 
def zscore_iqr_baseline(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """Reproduce the simple baseline from the EDA notebook."""
    z_anom = df["z_score"].abs() > z_thresh
    q1, q3 = df["return_1d"].quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_anom = (df["return_1d"] < q1 - 1.5 * iqr) | (df["return_1d"] > q3 + 1.5 * iqr)
    baseline = z_anom | iqr_anom
    return pd.DataFrame(
        {"baseline_score": baseline.astype(float), "baseline_anomaly": baseline},
        index=df.index,
    )
 
 
# ── Ensemble ──────────────────────────────────────────────────────────────────
 
SCORE_COLS = ["if_score", "lof_score", "dbscan_score", "baseline_score"]
FLAG_COLS  = ["if_anomaly", "lof_anomaly", "dbscan_anomaly", "baseline_anomaly"]
 
 
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0
 
 
def ensemble(df_scores: pd.DataFrame, vote_threshold: int = 2) -> pd.DataFrame:
    """
    Combine individual detector outputs into an ensemble.
 
    Args:
        df_scores:       DataFrame containing all score + flag columns.
        vote_threshold:  Min number of detectors that must agree to flag
                         the point as an anomaly (default 2 out of 4).
 
    Returns:
        df_scores with added columns:
            ensemble_score   – mean of min-max normalised individual scores (0-1)
            vote_count       – how many detectors flagged each point
            ensemble_anomaly – True if vote_count >= vote_threshold
    """
    # Normalise each score to [0, 1] before averaging
    norm = pd.concat(
        [_minmax(df_scores[c]).rename(c) for c in SCORE_COLS], axis=1
    )
    df_scores["ensemble_score"]   = norm.mean(axis=1)
    df_scores["vote_count"]       = df_scores[FLAG_COLS].sum(axis=1)
    df_scores["ensemble_anomaly"] = df_scores["vote_count"] >= vote_threshold
    return df_scores
 
 
# ── Main pipeline ─────────────────────────────────────────────────────────────
 
def detect(
    ticker: str,
    contamination: float = 0.05,
    vote_threshold: int = 2,
    force_preprocess: bool = False,
) -> pd.DataFrame:
    """
    Full detection pipeline for one ticker.
 
    Steps:
      1. Load (or re-generate) processed features.
      2. Run all four detectors.
      3. Build ensemble.
      4. Save results to data/processed/<TICKER>_anomalies.csv.
 
    Returns:
        DataFrame with all original columns + detector scores + flags.
    """
    # ── 1. Load features ──────────────────────────────────────────────────────
    proc_path = PROCESSED_DIR / f"{ticker.upper()}_processed.csv"
    if force_preprocess or not proc_path.exists():
        print(f"[detect] Running preprocess for {ticker}…")
        df = preprocess(ticker)
    else:
        df = pd.read_csv(proc_path, index_col="Date", parse_dates=True)
 
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Re-run preprocess.")
 
    # ── 2. Run detectors ──────────────────────────────────────────────────────
    print(f"[detect] Running detectors on {ticker} ({len(df)} rows)…")
    results = pd.concat(
        [
            isolation_forest(df, contamination=contamination),
            local_outlier_factor(df, contamination=contamination),
            dbscan_detector(df),
            zscore_iqr_baseline(df),
        ],
        axis=1,
    )
 
    # ── 3. Ensemble ───────────────────────────────────────────────────────────
    results = ensemble(results, vote_threshold=vote_threshold)
 
    # ── 4. Merge + save ───────────────────────────────────────────────────────
    out = pd.concat([df, results], axis=1)
    out_path = PROCESSED_DIR / f"{ticker.upper()}_anomalies.csv"
    out.to_csv(out_path)
 
    n_anom = results["ensemble_anomaly"].sum()
    pct    = n_anom / len(out) * 100
    print(f"[detect] Ensemble anomalies: {n_anom}/{len(out)} ({pct:.1f}%) → {out_path}")
    return out
 
 
def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table of the detected anomaly dates with key metrics."""
    cols = ["Close", "return_1d", "volume_ratio", "ensemble_score", "vote_count"] + FLAG_COLS
    cols = [c for c in cols if c in df.columns]
    return (
        df.loc[df["ensemble_anomaly"], cols]
        .sort_values("ensemble_score", ascending=False)
        .round(4)
    )
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Detect stock anomalies.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL TSLA")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--votes", type=int, default=2, help="Min detector votes for ensemble flag")
    parser.add_argument("--reprocess", action="store_true", help="Force re-preprocessing")
    args = parser.parse_args()
 
    for ticker in args.tickers:
        df_out = detect(ticker, contamination=args.contamination,
                        vote_threshold=args.votes, force_preprocess=args.reprocess)
        print(summary(df_out).to_string())
        print()
 
