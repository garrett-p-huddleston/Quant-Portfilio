from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

REGIME_LABELS = ("Low", "Normal", "High", "Extreme")


def compute_rolling_vol(
    returns: pd.Series,
    window: int,
    annual_days: int = 252,
) -> pd.Series:
    """Annualized rolling realized volatility from daily log returns."""
    return returns.rolling(window).std() * np.sqrt(annual_days)


def classify_regime(vol: float, config: Config) -> str:
    if vol < config.low_vol_threshold:
        return "Low"
    if vol < config.normal_vol_upper:
        return "Normal"
    if vol < config.high_vol_upper:
        return "High"
    return "Extreme"


def build_vol_series(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Append vol, clamped vol, and regime columns; drop warm-up rows."""
    out = df.copy()
    out["vol_raw"] = compute_rolling_vol(
        out["log_return"], config.vol_window, config.annual_trading_days
    )
    out["vol"] = out["vol_raw"].clip(lower=config.vol_floor, upper=config.vol_ceiling)
    out["regime"] = out["vol"].apply(
        lambda v: classify_regime(v, config) if pd.notna(v) else np.nan
    )
    return out.dropna(subset=["vol"]).copy()
