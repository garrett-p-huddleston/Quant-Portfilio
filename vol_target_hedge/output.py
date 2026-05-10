from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResults
from .config import Config

REGIME_COLORS: dict[str, str] = {
    "Low": "#2ecc71",
    "Normal": "#3498db",
    "High": "#e67e22",
    "Extreme": "#e74c3c",
}


# ── CSV export ────────────────────────────────────────────────────────────────

def save_signals_csv(df: pd.DataFrame, path: str | Path) -> None:
    cols = [
        "Close", "log_return", "vol_raw", "vol", "regime",
        "position", "adjustment", "rebalance", "reason",
    ]
    export = df[[c for c in cols if c in df.columns]].copy()
    export.index.name = "Date"
    export.to_csv(path)


# ── Console output ────────────────────────────────────────────────────────────

def print_backtest_summary(results: BacktestResults, config: Config) -> None:
    fd = results.full_df
    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS — Volatility Targeting Hedge")
    print("=" * 65)
    print(f"  Period         {fd.index[0].date()} → {fd.index[-1].date()}  ({len(fd)} trading days)")
    print(f"  Target notional  ${config.target_notional:>12,.0f}")
    print(f"  Vol window       {config.vol_window} days")
    print("-" * 65)
    print(f"  Total P&L          ${results.total_pnl:>13,.0f}")
    print(f"  Ann. $ return      ${results.annualized_dollar_return:>13,.0f} / year")
    print(f"  Sharpe ratio       {results.sharpe:>14.3f}")
    print(f"  Max drawdown       ${results.max_drawdown:>13,.0f}")
    print(f"  Win rate           {results.win_rate:>13.1%}")
    print(f"  Avg daily turnover {results.avg_daily_turnover:>10.1f} contracts")
    print("-" * 65)
    if results.folds:
        sharpes = [f.sharpe for f in results.folds]
        pct_pos = np.mean([f.total_pnl > 0 for f in results.folds])
        print(f"  Walk-forward folds   {len(results.folds)}")
        print(f"  Fold Sharpe          {min(sharpes):.3f}  to  {max(sharpes):.3f}  (mean {np.mean(sharpes):.3f})")
        print(f"  % folds profitable   {pct_pos:.0%}")
    print("=" * 65)


def print_explainer(df: pd.DataFrame, config: Config, n_examples: int = 6) -> None:
    print("\n" + "=" * 65)
    print("  POSITION SIZING EXPLAINER")
    print("=" * 65)
    print(
        f"  Formula:  N = ${config.target_notional:,.0f} / "
        f"(annual_vol × spot_price × ${config.contract_multiplier:.0f})"
    )
    print(
        "  Rationale: keeps annualized $ P&L volatility constant at "
        f"${config.target_notional:,.0f}\n"
        "             regardless of market regime.\n"
    )

    rebalance_rows = df[df.get("rebalance", False) & (df["reason"] != "")].head(n_examples)
    for date, row in rebalance_rows.iterrows():
        print(f"  [{date.date()}]  {row['reason']}")
        print()


def print_fold_table(results: BacktestResults) -> None:
    if not results.folds:
        return
    print("\n" + "=" * 85)
    print("  WALK-FORWARD FOLD SUMMARY")
    print("=" * 85)
    header = f"  {'Fold':>4}  {'Test Period':>22}  {'P&L ($)':>12}  {'Sharpe':>7}  {'Max DD ($)':>12}  {'Win%':>6}"
    print(header)
    print("-" * 85)
    for f in results.folds:
        period = f"{f.test_start.date()} – {f.test_end.date()}"
        print(
            f"  {f.fold_id:>4}  {period:>22}  "
            f"${f.total_pnl:>11,.0f}  {f.sharpe:>7.3f}  "
            f"${f.max_drawdown:>11,.0f}  {f.win_rate:>5.1%}"
        )
    print("=" * 85)


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_volatility(
    df: pd.DataFrame,
    config: Config,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["vol"] * 100, linewidth=1.2, color="#2c3e50", label="Realized Vol (%)")
    _shade_regimes(ax, df)

    for thresh, label in [
        (config.low_vol_threshold, "12% (Low/Normal)"),
        (config.normal_vol_upper, "20% (Normal/High)"),
        (config.high_vol_upper, "50% (High/Extreme)"),
    ]:
        ax.axhline(thresh * 100, linestyle="--", linewidth=0.8, alpha=0.6, color="#7f8c8d", label=label)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r)
        for r in ("Low", "Normal", "High", "Extreme")
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8, ncol=4)
    ax.set_title(
        f"Rolling {config.vol_window}-Day Realized Volatility (Annualized)", fontsize=13
    )
    ax.set_ylabel("Volatility (%)")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_position_sizes(
    df: pd.DataFrame,
    config: Config,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(df.index, df["position"].fillna(0), where="post", linewidth=1.2, color="#8e44ad")
    _shade_regimes(ax, df)
    ax.set_title(
        f"Short Position Size — Target Notional ${config.target_notional:,.0f}", fontsize=13
    )
    ax.set_ylabel("# Contracts (Short)")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_pnl(
    full_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        full_df.index,
        full_df["cum_net_pnl"] / 1_000,
        label="Vol-Target Strategy",
        linewidth=1.5,
        color="#27ae60",
    )
    ax.plot(
        full_df.index,
        full_df["cum_bnh_pnl"] / 1_000,
        label="Static Short Baseline",
        linewidth=1.0,
        color="#95a5a6",
        linestyle="--",
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_title("Cumulative P&L: Vol-Target vs. Static Short Baseline ($000s)", fontsize=13)
    ax.set_ylabel("Cumulative P&L ($000s)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_drawdown(
    full_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    cum = full_df["cum_net_pnl"]
    drawdown = (cum - cum.cummax()) / 1_000  # $000s

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(full_df.index, drawdown, 0, color="#e74c3c", alpha=0.7, label="Drawdown")
    ax.set_title("Strategy Drawdown ($000s)", fontsize=13)
    ax.set_ylabel("Drawdown ($000s)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_all(
    signals_df: pd.DataFrame,
    results: BacktestResults,
    config: Config,
    out_dir: Path,
) -> None:
    plot_volatility(signals_df, config, save_path=out_dir / "vol_chart.png")
    plot_position_sizes(signals_df, config, save_path=out_dir / "position_chart.png")
    plot_pnl(results.full_df, save_path=out_dir / "pnl_chart.png")
    plot_drawdown(results.full_df, save_path=out_dir / "drawdown_chart.png")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_regimes(ax: plt.Axes, df: pd.DataFrame) -> None:
    if "regime" not in df.columns:
        return
    prev_regime: Optional[str] = None
    start_date = None

    for date, row in df.iterrows():
        regime = row.get("regime")
        if pd.isna(regime):
            continue
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                ax.axvspan(start_date, date, alpha=0.08, color=REGIME_COLORS.get(prev_regime, "#cccccc"))
            start_date = date
            prev_regime = regime

    if prev_regime is not None and start_date is not None:
        ax.axvspan(start_date, df.index[-1], alpha=0.08, color=REGIME_COLORS.get(prev_regime, "#cccccc"))


def _save_or_show(fig: plt.Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
