from __future__ import annotations

import copy
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from .backtest import BacktestResults
from .config import Config


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_analysis(
    raw_df: pd.DataFrame,
    config: Config,
    full_pipeline_fn: Callable[[pd.DataFrame, Config], BacktestResults],
) -> Dict[str, Any]:
    """Run the full pipeline under varying target notionals and vol windows.

    full_pipeline_fn(raw_df, cfg) must return a BacktestResults.
    """
    notional_results: Dict[float, Dict] = {}
    for notional in config.sensitivity_notionals:
        cfg = copy.deepcopy(config)
        cfg.target_notional = notional
        r = full_pipeline_fn(raw_df, cfg)
        notional_results[notional] = _extract_metrics(r)

    window_results: Dict[int, Dict] = {}
    for window in config.sensitivity_windows:
        cfg = copy.deepcopy(config)
        cfg.vol_window = window
        r = full_pipeline_fn(raw_df, cfg)
        window_results[window] = _extract_metrics(r)

    return {"notional": notional_results, "window": window_results}


def _extract_metrics(r: BacktestResults) -> Dict[str, float]:
    return {
        "total_pnl": r.total_pnl,
        "annualized_dollar_return": r.annualized_dollar_return,
        "sharpe": r.sharpe,
        "max_drawdown": r.max_drawdown,
        "win_rate": r.win_rate,
        "avg_turnover": r.avg_daily_turnover,
    }


# ── Console reports ───────────────────────────────────────────────────────────

def print_sensitivity_report(sens: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 72)

    print("\n  Varying Target Notional  (vol_window held constant)")
    _header = f"  {'Notional':>12}  {'Total P&L':>13}  {'Sharpe':>7}  {'Max DD':>13}  {'Win%':>6}  {'Turnover':>9}"
    print(_header)
    print("  " + "-" * 70)
    for notional, m in sorted(sens["notional"].items()):
        print(
            f"  ${notional:>11,.0f}  ${m['total_pnl']:>12,.0f}  "
            f"{m['sharpe']:>7.3f}  ${m['max_drawdown']:>12,.0f}  "
            f"{m['win_rate']:>5.1%}  {m['avg_turnover']:>8.1f}"
        )

    print("\n  Varying Vol Lookback Window  (notional held constant)")
    _header2 = f"  {'Window':>8}  {'Total P&L':>13}  {'Sharpe':>7}  {'Max DD':>13}  {'Win%':>6}  {'Turnover':>9}"
    print(_header2)
    print("  " + "-" * 65)
    for window, m in sorted(sens["window"].items()):
        print(
            f"  {window:>7}d  ${m['total_pnl']:>12,.0f}  "
            f"{m['sharpe']:>7.3f}  ${m['max_drawdown']:>12,.0f}  "
            f"{m['win_rate']:>5.1%}  {m['avg_turnover']:>8.1f}"
        )
    print("=" * 72)


def print_regime_attribution(results: BacktestResults) -> None:
    descriptions = {
        "Low":     "Vol <12%    — hedge costs premium; carry is negative",
        "Normal":  "Vol 12-20%  — balanced; moderate hedge effectiveness",
        "High":    "Vol 20-50%  — hedge pays; equity losses offset by shorts",
        "Extreme": "Vol >50%    — maximum payout; crisis regime",
    }
    print("\n" + "=" * 72)
    print("  P&L ATTRIBUTION BY VOLATILITY REGIME")
    print("=" * 72)
    print(f"  {'Regime':<10}  {'P&L ($)':>13}  Description")
    print("  " + "-" * 68)
    for regime in ("Low", "Normal", "High", "Extreme"):
        pnl = results.regime_pnl.get(regime, 0.0)
        desc = descriptions.get(regime, "")
        print(f"  {regime:<10}  ${pnl:>12,.0f}  {desc}")
    print("=" * 72)
