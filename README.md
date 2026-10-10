# quanttro

An open-source Python toolkit for quantitative research: data fetching,
feature engineering, walk-forward backtesting, position sizing, portfolio
optimization, backtest-robustness testing (CPCV/PBO), a broker-agnostic
paper-trading bridge, and prop-firm challenge simulation, all in one
package. Works with any `yfinance` ticker — equities, crypto, gold, FX —
with first-class, built-in support for Borsa Istanbul (BIST), where this
project originated.

> **Disclaimer:** This project is for educational and research purposes
> only. Nothing in this repository — code, documentation, example output,
> or generated reports — is investment, trading, or financial advice. Past
> backtest performance does not predict future results. Data comes from
> `yfinance` and is not certified for trading decisions (see
> [`docs/data.md`](docs/data.md) for known data-quality caveats). Use at
> your own risk; the authors accept no liability for financial losses
> arising from use of this software.

This is a **research toolkit**, not a signal-selling product. It grew out of
a real research project that tested popular trading folklore (volume
effects, calendar effects, cross-asset lag, limit-up streaks) against real
BIST, crypto, and gold data — and documented what held up and what didn't.
Every pattern that survived that process (and several that didn't, as
cautionary examples) was turned into a reusable, tested function. The full
list of methodological lessons learned along the way lives in
[`docs/KNOWN_PITFALLS.md`](docs/KNOWN_PITFALLS.md) — read it before trusting
any diagnostic output, especially the limit-streak and autocorrelation
findings, which are easy to mistake for "predictability" when they're
actually signs of a manipulation/distress event.

## Install

```bash
pip install quanttro
```

To develop against this repo locally (editable install):

```bash
git clone https://github.com/HaasEnjoyer/quanttro.git && cd quanttro
python -m venv venv && source venv/bin/activate
pip install -e .
```

Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `yfinance`,
`lightgbm` (optional: `xgboost`).

## Quick start

```python
import quanttro as bq

# 1. Fetch — any yfinance ticker works, not just BIST (crypto/gold/FX too)
df = bq.fetch("THYAO.IS")              # single symbol -> DataFrame
data = bq.fetch("BIST30")              # whole index -> {ticker: DataFrame}

# 2. Features — lagged returns, RSI/MACD/Bollinger, calendar effects, all in one call
feats = bq.make_features(df)

# 3. Data quality check — catches real yfinance glitches (e.g. a 2005 redenomination
#    artifact in THYAO's price history) before they corrupt a backtest
issues = bq.check(df)

# 4. Walk-forward validation + a model
splitter = bq.WalkForwardSplitter(n_folds=5)
model = bq.make_model(task="regression")  # lightgbm / xgboost / random_forest / linear

# 5. Position sizing, portfolio construction, robustness testing
sizer = bq.PositionSizer(method="kelly", win_rate=0.55, win_loss_ratio=1.2)
weights = bq.hrp_optimize(returns_df)                      # Hierarchical Risk Parity
pbo = bq.probability_of_backtest_overfitting(trial_results)  # is this overfit?

# 6. Full performance report
report_html = bq.tearsheet(strategy_returns, benchmark_returns=bist100_returns)
```

Run the full, commented end-to-end walkthrough:

```bash
python examples/quickstart.py
```

## Modules

The top-level `quanttro` package is a convenience facade — `bq.fetch()`,
`bq.make_features()`, `bq.check()`, `bq.report()`, and the most commonly
used classes/functions from every submodule are all available directly on
`quanttro`. For the full API (including less common functions), import
the submodule directly.

| Submodule | What it's for | Docs |
|---|---|---|
| `quanttro.data` | Fetching OHLCV/macro data (any yfinance ticker, not just BIST), a parquet-based disk cache, and data-quality validators that catch real issues (price jumps, zero-volume runs, stale prices) | [`docs/data.md`](docs/data.md) |
| `quanttro.features` | Core price/volume features (lagged returns, volatility, ATR, limit-up/down flags), classic technical indicators (RSI, MACD, Bollinger, Stochastic, OBV), calendar effects, and cross-sectional (multi-stock) ranking | [`docs/features.md`](docs/features.md) |
| `quanttro.backtest` | Walk-forward splitting (expanding/rolling, with purge gaps), a model factory (LightGBM/XGBoost/RandomForest/Linear), 20 risk/performance metrics (Sharpe through Deflated Sharpe, VaR/CVaR, Kelly, Ulcer Index...), and cross-sectional ranking evaluation (Information Coefficient, tertile spread) | [`docs/backtest.md`](docs/backtest.md) |
| `quanttro.risk` | Position sizing (fixed-fractional, Kelly, volatility-target, ATR-based), dynamic drawdown throttling (single-strategy and multi-strategy/portfolio-level aggregation), stop-loss/take-profit simulation, transaction-cost/market-impact models, direction-aware slippage, and partial-fill/order-queuing simulation | [`docs/risk.md`](docs/risk.md) |
| `quanttro.portfolio` | Mean-variance optimization, Hierarchical Risk Parity, Ledoit-Wolf covariance shrinkage, discrete (whole-lot) allocation, and multi-strategy comparison | [`docs/portfolio.md`](docs/portfolio.md) |
| `quanttro.robustness` | Combinatorial Purged Cross-Validation, Probability of Backtest Overfitting (CSCV), Monte Carlo return resampling, and parameter sensitivity analysis — the backtest-robustness tools that mainstream open-source libraries (vectorbt, backtrader, zipline) don't ship | [`docs/robustness.md`](docs/robustness.md) |
| `quanttro.diagnostics` + `quanttro.reports` | Automatic anomaly detection (limit-up/down streaks, regime breaks) and report generation (symbol reports, full QuantStats-style tearsheets) | [`docs/diagnostics_and_reports.md`](docs/diagnostics_and_reports.md) |
| `quanttro.live` | A broker-agnostic bridge (`BrokerInterface`) between a strategy's signal and order placement — `SimulatedBroker` for realistic paper trading (uses `risk.execution`'s cost/slippage/partial-fill models) and `LiveTradingLoop` to turn target positions into orders. Connecting a real broker means implementing `BrokerInterface` for its API; no live BIST broker integration ships with this library. | [`docs/live.md`](docs/live.md) |
| `quanttro.propfirm` | Simulates a prop-firm ("funded trader") evaluation challenge against a real historical return series — daily loss limit (or none, for firms that only check overall drawdown), static/trailing max drawdown, profit target, and a simplified consistency-rule check — to answer "would this strategy have passed?" from actual history instead of a synthetic Monte Carlo guess. No real money, orders, or broker/firm integration involved; FTMO/Topstep are trademarks of their respective owners, used descriptively only, and example rule presets must be verified against each firm's current published terms. | [`docs/propfirm.md`](docs/propfirm.md) |

## Why `quanttro.robustness` exists

Backtest overfitting is the single most common reason a strategy that looks
great historically loses money live. Academic tools for detecting it
(Combinatorial Purged Cross-Validation and the Probability of Backtest
Overfitting, both from Lopez de Prado's work) are well established in the
literature but essentially absent from maintained open-source backtesting
libraries. `quanttro.robustness` implements them directly: run it on a
batch of random, signal-free strategies and it correctly reports a high
PBO (≈0.65 in our own test); run it on a batch where one strategy has a
genuine, consistent edge and PBO drops sharply (≈0.12). See
[`docs/robustness.md`](docs/robustness.md) for the full walkthrough.

## Repository layout

```
quanttro/
  data/          # fetching, caching, validation
  features/      # core, technical, calendar, cross-sectional features
  backtest/      # walk-forward splitting, models, metrics, ranking
  risk/          # position sizing, drawdown throttling, execution/cost models
  portfolio/     # mean-variance, HRP, discrete allocation, comparison
  robustness/    # CPCV, PBO, Monte Carlo, parameter sensitivity
  diagnostics/   # anomaly + regime-break detection
  reports/       # symbol reports and tearsheets
  live/          # broker interface, simulated (paper) broker, trading loop
  propfirm/      # prop-firm evaluation-challenge rule simulation
  cli.py         # `quanttro fetch` / `quanttro validate`
examples/
  quickstart.py  # end-to-end walkthrough (THYAO)
docs/            # one file per submodule, every example actually executed
tests/           # 277 tests, one file per submodule
```

Root-level `analyze_*.py` and `validate_*.py` scripts are the original,
one-off exploratory analyses that preceded this package — the patterns they
found (BIST volume/calendar effects, the KONTR limit-streak case, the THYAO
2005 data glitch) are what the `diagnostics` module now detects
automatically.

## A note on expectations

This library will not hand you a profitable strategy. The research that
produced it found, repeatedly and across asset classes (BIST equities,
PAXG/gold lag, BTC/altcoin lead-lag), that most intuitive "obvious" patterns
either don't survive out-of-sample validation or don't survive realistic
transaction costs once they do. What it *will* do is make it much harder to
fool yourself while you check — which, per the backtest-overfitting
literature this library leans on, is most of the actual work.
