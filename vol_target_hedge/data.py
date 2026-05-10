from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Required columns: Date, Close.  Volume and Open are used if present.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"Date", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return _validate_and_prepare(df)


def fetch_yfinance(
    ticker: str,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download daily OHLCV from yfinance and return a clean DataFrame."""
    logger.info("Fetching %s  %s → %s", ticker, start, end or "today")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'")

    # yfinance may return MultiIndex columns when a single ticker is fetched
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
    return _validate_and_prepare(raw[keep].copy())


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()

    n_missing = df["Close"].isna().sum()
    if n_missing > 0:
        logger.warning("Forward-filling %d missing Close values", n_missing)
        df["Close"] = df["Close"].ffill()

    # Warn about large calendar gaps (exchange closures create ~3-day gaps)
    gaps = df.index.to_series().diff().dt.days
    large_gaps = gaps[gaps > 7]
    if not large_gaps.empty:
        logger.warning(
            "%d date gap(s) > 7 calendar days — check for data integrity issues",
            len(large_gaps),
        )

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} rows of data — need at least 30 to compute anything useful"
        )

    # Remove any rows with non-positive or clearly erroneous prices
    bad_prices = df["Close"] <= 0
    if bad_prices.any():
        logger.warning("Dropping %d rows with non-positive Close prices", bad_prices.sum())
        df = df[~bad_prices]

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["log_return"])

    return df
