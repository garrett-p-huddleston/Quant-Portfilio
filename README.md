# Volatility Targeting Hedge

A production-grade Python implementation of a volatility targeting strategy for hedging long equity exposure via ES (S&P 500 e-mini) futures or any index future. Position sizes are dynamically adjusted to maintain constant annualized dollar P&L volatility across market regimes.

---

## Core Formula

```
N_contracts = Target_Notional / (Annual_Vol × Spot_Price × Contract_Multiplier)
```

This ensures:

```
N × annual_vol × spot × multiplier = Target_Notional   (annualized $ P&L vol)
```

**Example** at ES = \$4,500, vol = 28%, target = \$1M:

```
N = $1,000,000 / (0.28 × $4,500 × $50) = 15 contracts
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic ES data for offline testing
python generate_test_data.py

# Run with synthetic data
python main.py --csv data/es_synthetic.csv

# Fetch live ES futures via yfinance (requires internet)
python main.py --start 2020-01-01

# Override parameters
python main.py --csv data/es_synthetic.csv --notional 2000000 --vol-window 10

# Skip slow steps
python main.py --csv data/es_synthetic.csv --no-sensitivity --no-charts
```

---

## Project Layout

```
vol_target_hedge/
  config.py      — Config dataclass (all tuneable parameters)
  data.py        — CSV loader + yfinance fetcher + validation
  volatility.py  — Rolling vol, regime classification
  sizing.py      — Position sizing formula + signal generation
  backtest.py    — Walk-forward P&L engine + metrics
  output.py      — CSV export, console reports, 4 charts
  analysis.py    — Sensitivity analysis + regime attribution
main.py          — CLI entry point
generate_test_data.py  — Synthetic regime-switching ES data
results/         — Generated outputs (CSV, charts)
```

---

## Volatility Regimes

| Regime  | Annualized Vol | Hedge behavior                              |
|---------|---------------|----------------------------------------------|
| Low     | < 12%         | Maximum contracts held; carry cost is highest |
| Normal  | 12–20%        | Balanced sizing; moderate hedge cost          |
| High    | 20–50%        | Reduced contracts; hedge value increases       |
| Extreme | > 50%         | Minimum contracts; maximum P&L in crash        |

---

## Risk Controls

| Control              | Value          | Purpose                              |
|----------------------|---------------|---------------------------------------|
| Vol floor            | 5% annualized | Prevents unbounded position at low vol |
| Vol ceiling          | 100%           | Caps position in crisis regimes        |
| Position cap         | 110% of floor-vol position | Hard leverage limit     |
| Rebalance threshold  | 0.5% vol Δ     | Filters noise-driven trading           |
| Slippage             | \$50 / contract | 1 tick (ES) per trade                 |
| Commission           | \$15 / contract | Round-trip                            |

---

## Outputs

| File                      | Contents                                          |
|---------------------------|---------------------------------------------------|
| `results/daily_signals.csv` | Date, Close, Vol, Regime, Position, Reason      |
| `results/vol_chart.png`   | Rolling vol with regime shading                   |
| `results/position_chart.png` | Contract count over time                       |
| `results/pnl_chart.png`   | Strategy vs. static short baseline (cumulative)   |
| `results/drawdown_chart.png` | Drawdown series                                |

---

## CLI Reference

```
python main.py [OPTIONS]

Data:
  --csv FILE          CSV with Date, Close columns
  --ticker TICKER     yfinance ticker (default: ES=F)
  --start YYYY-MM-DD  Start date (default: 2018-01-01)
  --end YYYY-MM-DD    End date (default: today)

Strategy:
  --notional FLOAT    Target notional $ (default: 1,000,000)
  --vol-window INT    Vol lookback window in days (default: 20)
  --multiplier FLOAT  Contract $ multiplier (default: 50 for ES)
  --vol-floor FLOAT   Min annualized vol (default: 0.05)
  --vol-ceiling FLOAT Max annualized vol (default: 1.00)

Output:
  --output-dir DIR    Output directory (default: results/)
  --no-sensitivity    Skip sensitivity analysis
  --no-charts         Skip chart generation
```

---

## Walk-Forward Validation

The backtest uses a rolling walk-forward scheme:

- **Train window:** 252 days (~1 year)
- **Test window:** 63 days (~1 quarter)
- **Step:** 21 days (~1 month)

Each fold evaluates out-of-sample: the strategy has zero look-ahead bias since it only uses data available at each signal date.

---

## Sensitivity Analysis

Automatically tests:
- **Target notional:** \$500k, \$1M, \$2M
- **Vol window:** 10, 20, 40 days

P&L, Sharpe, max drawdown, and win rate are reported for each configuration.
