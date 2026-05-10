from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    total_pnl: float
    annualized_dollar_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_daily_turnover: float
    regime_pnl: Dict[str, float]
    df: pd.DataFrame = field(repr=False)


@dataclass
class BacktestResults:
    folds: List[FoldResult]
    full_df: pd.DataFrame = field(repr=False)
    total_pnl: float = 0.0
    annualized_dollar_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_daily_turnover: float = 0.0
    regime_pnl: Dict[str, float] = field(default_factory=dict)


# ── Core P&L engine ───────────────────────────────────────────────────────────

def _attach_pnl(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute daily gross P&L, transaction costs, and buy-and-hold baseline.

    Position signal at close[t-1] → held from close[t-1] to close[t].
    Transaction costs charged when position changes (slippage + commission).
    Short position profits when price falls: pnl = N × (prev_close - close) × mult.
    """
    out = df.copy()
    closes = out["Close"].to_numpy(dtype=float)
    positions = out["position"].fillna(0).to_numpy(dtype=float)
    adjustments = out["adjustment"].fillna(0).to_numpy(dtype=float)
    n = len(out)

    gross_pnl = np.zeros(n)
    tx_cost = np.zeros(n)

    for i in range(1, n):
        prev_pos = positions[i - 1]
        gross_pnl[i] = prev_pos * (closes[i - 1] - closes[i]) * config.contract_multiplier
        tx_cost[i] = (
            abs(adjustments[i]) * (config.slippage_per_contract + config.commission_per_contract)
        )

    # Static short baseline: constant contract count = median of active positions
    active = positions[positions > 0]
    baseline_pos = int(np.median(active)) if len(active) > 0 else 1
    bnh_pnl = np.zeros(n)
    for i in range(1, n):
        bnh_pnl[i] = baseline_pos * (closes[i - 1] - closes[i]) * config.contract_multiplier

    out["gross_pnl"] = gross_pnl
    out["tx_cost"] = tx_cost
    out["net_pnl"] = gross_pnl - tx_cost
    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    out["bnh_pnl"] = bnh_pnl
    out["cum_bnh_pnl"] = out["bnh_pnl"].cumsum()
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(
    net_pnl: pd.Series,
    annual_days: int = 252,
) -> Tuple[float, float, float, float, float]:
    """Return (total_pnl, ann_dollar_return, sharpe, max_drawdown, win_rate)."""
    if len(net_pnl) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    total_pnl = float(net_pnl.sum())
    mean_daily = float(net_pnl.mean())
    std_daily = float(net_pnl.std())

    ann_dollar_return = mean_daily * annual_days
    sharpe = (mean_daily * annual_days) / (std_daily * np.sqrt(annual_days)) if std_daily > 0 else 0.0

    cum = net_pnl.cumsum()
    max_drawdown = float((cum - cum.cummax()).min())

    win_rate = float((net_pnl > 0).mean())

    return total_pnl, ann_dollar_return, sharpe, max_drawdown, win_rate


# ── Walk-forward engine ───────────────────────────────────────────────────────

def run_backtest(signals_df: pd.DataFrame, config: Config) -> BacktestResults:
    """Full walk-forward backtest on a pre-built signals DataFrame.

    Each fold:
        train = [start : start + train_window]   (not used for fitting here,
        test  = [start + train_window : +test_window]   present for regime context)

    Steps forward by step_size until the data is exhausted.
    """
    full_df = _attach_pnl(signals_df, config)

    n = len(full_df)
    folds: List[FoldResult] = []
    fold_id = 0
    idx = 0

    while idx + config.train_window + config.test_window <= n:
        train_slice = full_df.iloc[idx : idx + config.train_window]
        test_slice = full_df.iloc[idx + config.train_window : idx + config.train_window + config.test_window]

        net = test_slice["net_pnl"]
        total, ann_ret, sharpe, max_dd, win_rate = _compute_metrics(net, config.annual_trading_days)
        avg_turnover = float(test_slice["adjustment"].abs().mean())
        regime_pnl = _regime_pnl(test_slice)

        folds.append(
            FoldResult(
                fold_id=fold_id,
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                total_pnl=total,
                annualized_dollar_return=ann_ret,
                sharpe=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                avg_daily_turnover=avg_turnover,
                regime_pnl=regime_pnl,
                df=test_slice,
            )
        )

        idx += config.step_size
        fold_id += 1

    # Full-period aggregate
    all_net = full_df["net_pnl"]
    total, ann_ret, sharpe, max_dd, win_rate = _compute_metrics(all_net, config.annual_trading_days)

    return BacktestResults(
        folds=folds,
        full_df=full_df,
        total_pnl=total,
        annualized_dollar_return=ann_ret,
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_daily_turnover=float(full_df["adjustment"].abs().mean()),
        regime_pnl=_regime_pnl(full_df),
    )


def _regime_pnl(df: pd.DataFrame) -> Dict[str, float]:
    if "regime" not in df.columns:
        return {}
    return df.groupby("regime")["net_pnl"].sum().to_dict()
