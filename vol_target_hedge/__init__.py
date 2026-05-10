"""vol_target_hedge — Volatility Targeting Strategy for Futures Index Hedging."""

from .config import Config
from .data import fetch_yfinance, load_csv
from .volatility import build_vol_series
from .sizing import build_signals, position_size_from_vol
from .backtest import run_backtest, BacktestResults
from .output import (
    plot_all,
    plot_drawdown,
    plot_pnl,
    plot_position_sizes,
    plot_volatility,
    print_backtest_summary,
    print_explainer,
    print_fold_table,
    save_signals_csv,
)
from .analysis import (
    print_regime_attribution,
    print_sensitivity_report,
    sensitivity_analysis,
)

__all__ = [
    "Config",
    "fetch_yfinance",
    "load_csv",
    "build_vol_series",
    "build_signals",
    "position_size_from_vol",
    "run_backtest",
    "BacktestResults",
    "plot_all",
    "plot_drawdown",
    "plot_pnl",
    "plot_position_sizes",
    "plot_volatility",
    "print_backtest_summary",
    "print_explainer",
    "print_fold_table",
    "save_signals_csv",
    "print_regime_attribution",
    "print_sensitivity_report",
    "sensitivity_analysis",
]
