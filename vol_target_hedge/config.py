from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Position sizing ──────────────────────────────────────────────────────
    target_notional: float = 1_000_000.0
    vol_window: int = 20
    vol_floor: float = 0.05        # 5% annualized — prevents division explosion
    vol_ceiling: float = 1.00      # 100% annualized — caps crisis-regime sizing
    position_cap_pct: float = 1.10 # max position notional = 110% of target
    rebalance_threshold: float = 0.005  # 0.5% absolute vol change triggers trade

    # ── Contract spec (ES default) ───────────────────────────────────────────
    contract_multiplier: float = 50.0    # $50 per index point
    slippage_per_contract: float = 50.0  # 1 tick ($12.50) × 4 fills ≈ $50
    commission_per_contract: float = 15.0

    # ── Calendar ─────────────────────────────────────────────────────────────
    annual_trading_days: int = 252

    # ── Walk-forward parameters ──────────────────────────────────────────────
    train_window: int = 252   # ~1 year
    test_window: int = 63     # ~1 quarter
    step_size: int = 21       # ~1 month

    # ── Volatility regime thresholds (annualized) ────────────────────────────
    low_vol_threshold: float = 0.12   # < 12%
    normal_vol_upper: float = 0.20    # 12–20%
    high_vol_upper: float = 0.50      # 20–50%

    # ── Data source ──────────────────────────────────────────────────────────
    ticker: str = "ES=F"

    # ── Sensitivity analysis parameters ──────────────────────────────────────
    sensitivity_notionals: List[float] = field(
        default_factory=lambda: [500_000.0, 1_000_000.0, 2_000_000.0]
    )
    sensitivity_windows: List[int] = field(
        default_factory=lambda: [10, 20, 40]
    )
