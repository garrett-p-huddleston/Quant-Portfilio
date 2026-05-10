from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import Config


def position_size_from_vol(
    vol_annual: float,
    spot_price: float,
    config: Config,
) -> int:
    """Number of short contracts to hold given current annualized vol and spot.

    Formula: N = Target_Notional / (annualized_vol × spot × contract_multiplier)

    This ensures the annualized dollar P&L volatility of the position equals
    target_notional regardless of market regime:
        N × annualized_vol × spot × multiplier = target_notional
    """
    vol = np.clip(vol_annual, config.vol_floor, config.vol_ceiling)
    raw_contracts = config.target_notional / (vol * spot_price * config.contract_multiplier)

    # Cap: never exceed 110% of the position you'd hold at the vol floor.
    # This bounds maximum leverage when vol is at its absolute minimum.
    max_contracts = (
        config.target_notional * config.position_cap_pct
    ) / (config.vol_floor * spot_price * config.contract_multiplier)

    return max(0, int(min(raw_contracts, max_contracts)))


def format_sizing_math(vol: float, spot: float, contracts: int, config: Config) -> str:
    """One-line plain-English formula string for a given position."""
    vol_c = np.clip(vol, config.vol_floor, config.vol_ceiling)
    denom = vol_c * spot * config.contract_multiplier
    return (
        f"${config.target_notional:,.0f} / "
        f"({vol_c:.4f} × ${spot:,.2f} × ${config.contract_multiplier:.0f}) "
        f"= {contracts} contracts"
    )


def build_signals(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Walk through the vol series and produce daily position signals.

    Returns input DataFrame with added columns:
        position   — number of short contracts to hold entering the next day
        adjustment — change vs. prior day's position (+buy, -sell)
        rebalance  — True when a trade is triggered
        reason     — plain-English explanation (non-empty only on rebalance days)
    """
    out = df.copy()
    positions: list[int | float] = []
    adjustments: list[int] = []
    rebalance_flags: list[bool] = []
    reasons: list[str] = []

    prev_pos: Optional[int] = None
    prev_vol: Optional[float] = None

    for _, row in out.iterrows():
        if pd.isna(row["vol"]):
            positions.append(np.nan)
            adjustments.append(0)
            rebalance_flags.append(False)
            reasons.append("")
            continue

        target = position_size_from_vol(row["vol"], row["Close"], config)

        if prev_pos is None:
            rebalance = True
            reason = (
                f"Initial position — vol={row['vol']:.1%}, "
                f"spot=${row['Close']:,.2f}; "
                + format_sizing_math(row["vol"], row["Close"], target, config)
            )
            new_pos = target
        elif abs(row["vol"] - prev_vol) > config.rebalance_threshold:  # type: ignore[operator]
            direction = "spiked" if row["vol"] > prev_vol else "compressed"  # type: ignore[operator]
            delta = row["vol"] - prev_vol  # type: ignore[operator]
            rebalance = True
            reason = (
                f"Vol {direction} {prev_vol:.1%} → {row['vol']:.1%} "  # type: ignore[operator]
                f"(Δ={delta:+.1%}); "
                f"position {prev_pos} → {target} contracts; "
                + format_sizing_math(row["vol"], row["Close"], target, config)
            )
            new_pos = target
        else:
            rebalance = False
            reason = ""
            new_pos = prev_pos  # hold — vol hasn't moved enough to warrant a trade

        adj = new_pos - (prev_pos if prev_pos is not None else 0)
        positions.append(new_pos)
        adjustments.append(adj)
        rebalance_flags.append(rebalance)
        reasons.append(reason)

        prev_pos = new_pos
        prev_vol = row["vol"]

    out["position"] = positions
    out["adjustment"] = adjustments
    out["rebalance"] = rebalance_flags
    out["reason"] = reasons
    return out
