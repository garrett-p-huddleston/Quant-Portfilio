#!/usr/bin/env python3
"""Generate synthetic ES futures price series for offline testing.

Simulates a GBM with regime-switching volatility so all four vol regimes appear.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_es_data(
    start: str = "2018-01-01",
    days: int = 1762,   # ~7 years of trading days
    seed: int = 42,
    out_path: Path = Path("data/es_synthetic.csv"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(start=start, periods=days)

    # Regime-switching vol (Markov chain: Low / Normal / High / Extreme)
    regime_vols = [0.09, 0.16, 0.30, 0.65]     # annualized
    regime_names = ["Low", "Normal", "High", "Extreme"]
    # Transition matrix rows = from, cols = to
    # Tuned so all four regimes appear in ~1762 day dataset
    transition = np.array([
        [0.93, 0.05, 0.02, 0.00],
        [0.05, 0.87, 0.07, 0.01],
        [0.05, 0.10, 0.73, 0.12],
        [0.10, 0.15, 0.30, 0.45],
    ])

    current_regime = 1  # start in Normal
    vol_series = np.empty(days)
    for i in range(days):
        vol_series[i] = regime_vols[current_regime]
        current_regime = rng.choice(4, p=transition[current_regime])

    # GBM: drift = 0.07/252, vol from regime
    dt = 1 / 252
    drift = 0.07 * dt
    daily_vols = vol_series * np.sqrt(dt)
    log_returns = drift - 0.5 * daily_vols ** 2 + daily_vols * rng.standard_normal(days)

    # Start at 2700 (roughly ES in early 2018)
    prices = 2700 * np.exp(np.cumsum(log_returns))
    prices[0] = 2700.0

    volume = rng.integers(800_000, 2_500_000, size=days)

    df = pd.DataFrame(
        {"Date": dates, "Close": np.round(prices, 2), "Volume": volume}
    )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    print(f"Price range: ${prices.min():,.2f} – ${prices.max():,.2f}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=1762)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/es_synthetic.csv"))
    args = p.parse_args()
    generate_es_data(days=args.days, seed=args.seed, out_path=args.out)
