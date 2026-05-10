#!/usr/bin/env python3
"""
Volatility Targeting Hedge — main entry point.

Usage examples
──────────────
  # Fetch ES futures via yfinance (default)
  python main.py

  # Supply a custom CSV (Date, Close columns required)
  python main.py --csv /path/to/data.csv

  # Override key parameters
  python main.py --notional 2000000 --vol-window 10 --start 2019-01-01

  # Skip sensitivity analysis (faster run)
  python main.py --no-sensitivity

  # Save outputs to a custom directory
  python main.py --output-dir /tmp/hedge_results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from vol_target_hedge import (
    Config,
    fetch_yfinance,
    load_csv,
    build_vol_series,
    build_signals,
    run_backtest,
    plot_all,
    print_backtest_summary,
    print_explainer,
    print_fold_table,
    save_signals_csv,
    sensitivity_analysis,
    print_sensitivity_report,
    print_regime_attribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Volatility Targeting Hedge Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("Data source (use one)")
    data.add_argument("--csv", type=Path, metavar="FILE",
                      help="CSV with Date, Close columns")
    data.add_argument("--ticker", default="ES=F",
                      help="yfinance ticker (default: ES=F)")
    data.add_argument("--start", default="2018-01-01",
                      help="Start date YYYY-MM-DD (default: 2018-01-01)")
    data.add_argument("--end", default=None,
                      help="End date YYYY-MM-DD (default: today)")

    strat = p.add_argument_group("Strategy parameters")
    strat.add_argument("--notional", type=float, default=1_000_000.0,
                       help="Target notional $ (default: 1,000,000)")
    strat.add_argument("--vol-window", type=int, default=20,
                       help="Vol lookback window in days (default: 20)")
    strat.add_argument("--multiplier", type=float, default=50.0,
                       help="Contract $ multiplier (default: 50 for ES)")
    strat.add_argument("--vol-floor", type=float, default=0.05,
                       help="Min annualized vol for sizing (default: 0.05)")
    strat.add_argument("--vol-ceiling", type=float, default=1.00,
                       help="Max annualized vol for sizing (default: 1.00)")

    out = p.add_argument_group("Output options")
    out.add_argument("--output-dir", type=Path, default=Path("results"),
                     help="Directory for saved files (default: results/)")
    out.add_argument("--no-sensitivity", action="store_true",
                     help="Skip sensitivity analysis (faster)")
    out.add_argument("--no-charts", action="store_true",
                     help="Skip chart generation")

    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _build_config(args: argparse.Namespace) -> Config:
    return Config(
        target_notional=args.notional,
        vol_window=args.vol_window,
        contract_multiplier=args.multiplier,
        vol_floor=args.vol_floor,
        vol_ceiling=args.vol_ceiling,
        ticker=args.ticker,
    )


def _full_pipeline(raw_df: pd.DataFrame, cfg: Config):
    """Build signals and run backtest; used by sensitivity analysis."""
    vol_df = build_vol_series(raw_df, cfg)
    sig_df = build_signals(vol_df, cfg)
    return run_backtest(sig_df, cfg)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _SYNTHETIC = Path("data/es_synthetic.csv")

    if args.csv:
        log.info("Loading data from %s", args.csv)
        raw_df = load_csv(args.csv)
    else:
        log.info("Fetching %s  %s → %s", config.ticker, args.start, args.end or "today")
        try:
            raw_df = fetch_yfinance(config.ticker, args.start, args.end)
        except Exception as exc:
            if _SYNTHETIC.exists():
                log.warning(
                    "yfinance failed (%s). Falling back to %s. "
                    "Pass --csv <file> to use your own data.",
                    exc,
                    _SYNTHETIC,
                )
                raw_df = load_csv(_SYNTHETIC)
            else:
                raise SystemExit(
                    f"yfinance failed ({exc}) and no fallback data found.\n"
                    f"Generate synthetic data with:  python generate_test_data.py\n"
                    f"Then run:  python main.py --csv data/es_synthetic.csv"
                ) from exc

    log.info(
        "Loaded %d rows  (%s → %s)",
        len(raw_df),
        raw_df.index[0].date(),
        raw_df.index[-1].date(),
    )

    # ── 2. Volatility engine ──────────────────────────────────────────────────
    vol_df = build_vol_series(raw_df, config)
    log.info(
        "Vol computed on %d rows  (warm-up removed %d rows)",
        len(vol_df),
        len(raw_df) - len(vol_df),
    )

    regime_counts = vol_df["regime"].value_counts().to_dict()
    log.info("Regime distribution: %s", regime_counts)

    # ── 3. Position signals ───────────────────────────────────────────────────
    signals_df = build_signals(vol_df, config)
    n_rebalances = int(signals_df["rebalance"].sum())
    log.info(
        "%d signals generated  |  %d rebalancing events  (%.1f%% of days)",
        len(signals_df),
        n_rebalances,
        100 * n_rebalances / len(signals_df),
    )

    # ── 4. Save signals CSV ───────────────────────────────────────────────────
    signals_path = out_dir / "daily_signals.csv"
    save_signals_csv(signals_df, signals_path)
    log.info("Signals saved → %s", signals_path)

    # ── 5. Explainability console output ──────────────────────────────────────
    print_explainer(signals_df, config)

    # ── 6. Backtest ───────────────────────────────────────────────────────────
    results = run_backtest(signals_df, config)

    print_backtest_summary(results, config)
    print_fold_table(results)
    print_regime_attribution(results)

    # ── 7. Charts ─────────────────────────────────────────────────────────────
    if not args.no_charts:
        plot_all(signals_df, results, config, out_dir)
        log.info("Charts saved → %s/", out_dir)
    else:
        log.info("Chart generation skipped (--no-charts)")

    # ── 8. Sensitivity analysis ───────────────────────────────────────────────
    if not args.no_sensitivity:
        log.info("Running sensitivity analysis (%d notionals × %d windows)…",
                 len(config.sensitivity_notionals), len(config.sensitivity_windows))
        sens = sensitivity_analysis(raw_df, config, _full_pipeline)
        print_sensitivity_report(sens)
    else:
        log.info("Sensitivity analysis skipped (--no-sensitivity)")

    log.info("Done.  All outputs in %s/", out_dir)


if __name__ == "__main__":
    main()
