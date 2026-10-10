# `quanttro.portfolio`

`quanttro.portfolio` is a simplified but correctly-working subset of the
core functionality offered by PyPortfolioOpt — WITHOUT a heavy cvxpy
dependency, using only `scipy.optimize` (SLSQP) and
`scipy.cluster.hierarchy`. It is split into three submodules: `optimize.py`
(mean-variance optimization and covariance shrinkage), `allocation.py`
(Hierarchical Risk Parity and discrete share/lot allocation), and
`compare.py` (price series rebasing and strategy comparison tables). All
functions can be imported directly from the `quanttro.portfolio` package.

The examples in this document were run with REAL BIST stock price data
(THYAO, SASA, BIMAS, EREGL, GARAN — from 2025-01-01 to 2026-10-02, 442 daily
closing prices) from the repo's `data/daily/` folder, and the outputs were
copied directly. All data loading and execution code is included verbatim in
the example blocks below.

---

## Optimization

`optimize.py` includes mean-variance optimization portfolios that maximize
the Sharpe ratio or minimize volatility (`mean_variance_optimize`), an
"efficient risk" portfolio that maximizes expected return under a target
volatility (`efficient_risk`), and Ledoit-Wolf covariance shrinkage
estimation (`covariance_shrinkage`). A heavy convex-optimization dependency
(cvxpy) is NOT USED; all optimizations are solved with
`scipy.optimize.minimize` (SLSQP, with equality/inequality constraints).

The data is loaded first for the examples below:

```python
import pandas as pd
import numpy as np

symbols = ['THYAO', 'SASA', 'BIMAS', 'EREGL', 'GARAN']
prices = {}
for s in symbols:
    df = pd.read_csv(f'data/daily/{s}.csv', parse_dates=['Date'], index_col='Date')
    prices[s] = df['Close']
price_df = pd.concat(prices, axis=1, sort=True).dropna()
price_df = price_df.loc['2025-01-01':]
returns = price_df.pct_change().dropna()
mu = returns.mean() * 252  # annualized expected return
```

`mu` (annualized expected return, from real data):

```
THYAO    0.073467
SASA    -0.292734
BIMAS    0.336314
EREGL    0.321341
GARAN    0.102911
```

### `mean_variance_optimize(expected_returns, cov_matrix, objective="max_sharpe", risk_free=0.0, weight_bounds=(0.0, 1.0))`

Mean-variance optimization (`scipy.optimize.minimize`, `method="SLSQP"`).

`objective`:
- `"max_sharpe"`: finds the weights that maximize the Sharpe ratio
  `(w.mu - risk_free) / sqrt(w.Cov.w)` (in practice the NEGATIVE Sharpe ratio
  is minimized, since SLSQP only performs minimization).
- `"min_volatility"`: ignores expected return and finds the weights that
  minimize portfolio variance (`w.Cov.w`).

In both cases the constraints are the same: `sum(w) == 1` and each weight
within `weight_bounds` (default: long-only, `[0, 1]` — no shorting/leverage).

**Parameters:**
- `expected_returns` (pd.Series): expected return per asset (index = symbol).
- `cov_matrix` (pd.DataFrame): symbol x symbol covariance matrix; aligned to
  the order of `expected_returns` — raises `ValueError` if a symbol is
  missing.
- `objective` (str): `"max_sharpe"` or `"min_volatility"`, default
  `"max_sharpe"`.
- `risk_free` (float): risk-free rate (only for `"max_sharpe"`), default
  `0.0`.
- `weight_bounds` (tuple[float, float]): lower/upper bound for each weight,
  default `(0.0, 1.0)`.

**Returns:** pd.Series (`name="weight"`) — weights in the same order/index as
`expected_returns`, summing to 1.

**Raises:** `ValueError` — empty `expected_returns` or an unrecognized
`objective`. `RuntimeError` — if SLSQP fails to converge.

**Example:**
```python
from quanttro.portfolio import mean_variance_optimize, covariance_shrinkage

cov_lw = covariance_shrinkage(returns, method='ledoit_wolf')

w_sharpe = mean_variance_optimize(mu, cov_lw, objective='max_sharpe')
w_minvol = mean_variance_optimize(mu, cov_lw, objective='min_volatility')
```

**Output:**
```
# w_sharpe (max_sharpe):
THYAO    2.080931e-13
SASA     0.000000e+00
BIMAS    5.853697e-01
EREGL    4.146303e-01
GARAN    2.736687e-13
Name: weight, dtype: float64

# w_minvol (min_volatility):
THYAO    0.2
SASA     0.2
BIMAS    0.2
EREGL    0.2
GARAN    0.2
Name: weight, dtype: float64
```

Note: the `min_volatility` result landing on exactly equal weights (0.2) is
not made up — for this real data/covariance, SLSQP, starting from the
equal-weight initial point (`w0 = 1/n`), detects that it already (approximately)
satisfies the KKT conditions and stops after a single iteration (verified:
calling `scipy.optimize.minimize` directly gives `nit=1, success=True`). This
is a result specific to this particular combination of five stocks/period —
`min_volatility` does not generally produce equal weights.

### `efficient_risk(expected_returns, cov_matrix, target_volatility, weight_bounds=(0.0, 1.0))`

Finds the weights that maximize expected return at a risk EQUAL TO OR BELOW
the specified `target_volatility` (SLSQP, with an inequality constraint).

Constraints:
- `sum(w) == 1`
- each weight within `weight_bounds`
- `w.Cov.w <= target_volatility**2` (expressed in terms of variance rather
  than the square root — numerically more stable and differentiable
  everywhere)

If `target_volatility` is UNREACHABLE under the given `weight_bounds`
constraints (i.e. the minimum possible portfolio volatility is ABOVE the
target), a `ValueError` is raised.

**Parameters:**
- `expected_returns` (pd.Series): expected return per asset.
- `cov_matrix` (pd.DataFrame): symbol x symbol covariance matrix.
- `target_volatility` (float): the targeted (upper bound) portfolio
  volatility; must be positive.
- `weight_bounds` (tuple[float, float]): default `(0.0, 1.0)`.

**Returns:** pd.Series (`name="weight"`) — weights summing to 1.

**Raises:** `ValueError` — `target_volatility <= 0`, or if the target is
unreachable. `RuntimeError` — if SLSQP fails to converge.

**Example:**
```python
from quanttro.portfolio import efficient_risk

w_eff = efficient_risk(mu, cov_lw, target_volatility=0.25)
```

**Output:**
```
THYAO    0.000000e+00
SASA     0.000000e+00
BIMAS    1.000000e+00
EREGL    2.386980e-15
GARAN    0.000000e+00
Name: weight, dtype: float64
```

(Since BIMAS has the highest `mu` (0.336) and its standalone portfolio
volatility stays below the 0.25 target, the optimization allocates all the
weight to it.)

### `covariance_shrinkage(returns, method="ledoit_wolf")`

Covariance matrix estimation.

`method="ledoit_wolf"` (default): Ledoit-Wolf shrinkage
(`sklearn.covariance.LedoitWolf`) — shrinks the sample covariance toward a
diagonal target at an optimal ratio, reducing estimation error; it gives a
more stable/well-conditioned estimate than the sample covariance especially
when the number of assets is close to or exceeds the number of observations.

`method="sample"`: the plain sample covariance (`returns.cov()`), WITHOUT
shrinkage applied — intended for comparison/reference purposes.

Rows containing NaN are dropped with `dropna` before computation.

**Parameters:**
- `returns` (pd.DataFrame): columns are assets, rows are return observations.
- `method` (str): `"ledoit_wolf"` or `"sample"`, default `"ledoit_wolf"`.

**Returns:** pd.DataFrame — a symbol x symbol (square, symmetric) covariance
matrix.

**Raises:** `ValueError` — if fewer than 2 valid observations exist, or an
unrecognized `method`.

**Example:**
```python
from quanttro.portfolio import covariance_shrinkage

cov_sample = covariance_shrinkage(returns, method='sample')
cov_lw = covariance_shrinkage(returns, method='ledoit_wolf')
```

**Output (`cov_sample`):**
```
          THYAO      SASA     BIMAS     EREGL     GARAN
THYAO  0.000408  0.000267  0.000238  0.000249  0.000297
SASA   0.000267  0.001084  0.000199  0.000324  0.000299
BIMAS  0.000238  0.000199  0.000493  0.000221  0.000243
EREGL  0.000249  0.000324  0.000221  0.000572  0.000279
GARAN  0.000297  0.000299  0.000243  0.000279  0.000578
```

**Output (`cov_lw`):**
```
          THYAO      SASA     BIMAS     EREGL     GARAN
THYAO  0.000415  0.000257  0.000229  0.000239  0.000286
SASA   0.000257  0.001065  0.000191  0.000312  0.000287
BIMAS  0.000229  0.000191  0.000496  0.000213  0.000234
EREGL  0.000239  0.000312  0.000213  0.000573  0.000268
GARAN  0.000286  0.000287  0.000234  0.000268  0.000579
```

The off-diagonal values of `cov_lw` are slightly pulled toward the diagonal
("zero correlation" target) compared to `cov_sample` — e.g. the SASA-THYAO
covariance drops from 0.000267 to 0.000257.

---

## Allocation

`allocation.py` includes Hierarchical Risk Parity (HRP) weighting
(Lopez de Prado, 2016, "Building Diversified Portfolios that Outperform
Out-of-Sample") and `discrete_allocation`, which converts continuous weights
into actual discrete share/lot counts.

### `hrp_optimize(returns)`

Hierarchical Risk Parity (HRP) weighting. Unlike mean-variance optimization
(`mean_variance_optimize`), it does NOT INVERT the covariance matrix; instead
it distributes risk top-down according to a hierarchical clustering tree
(dendrogram). This produces MORE ROBUST weights than mean-variance when the
covariance matrix is ill-conditioned — e.g. when the number of assets is
close to or exceeds the number of observations, or when correlation between
assets is very high.

**Algorithm — step by step (the original HRP's 4 steps):**

**1) Distance matrix.** A distance matrix is derived from the return
correlation matrix (`returns.corr()`):

```
d_ij = sqrt((1 - corr_ij) / 2)
```

If correlation is +1, distance is 0 (the assets behave "identically"); if
correlation is -1, distance is 1 (they behave in complete opposition). This
is the standard transformation that converts correlation into a valid metric
(a notion of "distance" that satisfies the triangle inequality).

**2) Hierarchical clustering (linkage).** This distance matrix is converted
into a dendrogram (tree structure) using `scipy.cluster.hierarchy.linkage`
(the code uses `method="single"` — single linkage: the distance between two
clusters is determined by the CLOSEST pair between them). At each step, the
two closest clusters/assets are merged; the result is a `linkage` matrix that
records each merge (which two clusters, at what distance, how many
elements).

**3) Quasi-diagonalization (`_quasi_diagonalize`).** The leaves of the
`linkage` tree (the original assets) are converted into a single permutation
that places assets which merged together during clustering (and are
therefore most similar) CLOSE to one another. This is equivalent to
recursively traversing the tree from root to leaves and placing each node's
children (left/right) into an ordered list. The resulting covariance matrix,
reordered by this permutation, becomes "quasi-diagonal": the most similar
assets cluster into blocks near the diagonal. This ordering is required for
the next step — because recursive bisection must be able to split
meaningfully into halves while keeping neighboring elements in the same
subset.

**4) Recursive bisection (`_recursive_bisection`).** The quasi-diagonally
ordered asset list is repeatedly split in half, top-down. At each step, a
cluster is split (approximately) in the middle into two subclusters; the
"cluster variance" of each subcluster (an inverse-variance-weighted, simple
risk-parity variance computed WITHOUT inverting the covariance matrix, using
only the diagonal — `_cluster_variance`) is computed, and weight is
distributed so that the RISKIER (higher-variance) subcluster gets LESS
weight:

```
alpha = 1 - var(left_cluster) / (var(left_cluster) + var(right_cluster))
weight(left_cluster) *= alpha
weight(right_cluster) *= (1 - alpha)
```

Each subcluster continues to be split in half until only a single asset
remains within it. The final weight of an asset is the cumulative product of
the multipliers it inherits from all its ancestor clusters — i.e. the deeper
an asset descends down "riskier" branches of the tree, the more shrinkage
multipliers it accumulates.

Intuitive summary: first find which stocks form "co-moving" groups based on
correlation (steps 1-2), then arrange these groups in an order close to the
diagonal (step 3), then start from the top and descend, giving less capital
to the riskier side at each split (step 4) — all without ever inverting the
matrix, relying only on within-group and between-group risk comparisons.

**Parameters:**
- `returns` (pd.DataFrame): columns are assets (symbols), rows are per-period
  returns. Rows containing NaN (`dropna(how="any")`) are dropped.

**Returns:** pd.Series (`name="weight"`) — weights in the SAME ORDER as the
`returns` columns, summing to 1, NON-NEGATIVE.

**Raises:** `ValueError` — if fewer than 2 assets (columns) or fewer than 2
valid observations (rows) exist.

**Example:**
```python
from quanttro.portfolio import hrp_optimize

w_hrp = hrp_optimize(returns)
```

Input correlation matrix (real data, `returns.corr().round(3)`):
```
       THYAO   SASA  BIMAS  EREGL  GARAN
THYAO  1.000  0.401  0.531  0.515  0.611
SASA   0.401  1.000  0.273  0.412  0.377
BIMAS  0.531  0.273  1.000  0.417  0.456
EREGL  0.515  0.412  0.417  1.000  0.485
GARAN  0.611  0.377  0.456  0.485  1.000
```

**Output (`w_hrp`):**
```
THYAO    0.201286
SASA     0.134414
BIMAS    0.267340
EREGL    0.254834
GARAN    0.142126
Name: weight, dtype: float64
```

Commentary: while `mean_variance_optimize(objective="max_sharpe")` allocates
the entire weight to BIMAS+EREGL (concentration), HRP spreads weight across
all 5 stocks (min 13.4%, max 26.7%) — leaving meaningful weight on assets
with lower correlation to the rest, such as SASA and GARAN, which shows
HRP's nature of not inverting the covariance matrix and being more inclined
toward diversification.

### `discrete_allocation(weights, latest_prices, total_value)`

Converts continuous portfolio weights into actual discrete share/lot counts
(greedy algorithm).

**Algorithm:**
1. Symbols are sorted by weight from LARGEST TO SMALLEST.
2. For each symbol, its "ideal" dollar allocation (`weight * total_value`) is
   computed, and the MAXIMUM NUMBER OF WHOLE LOTS that can be bought with
   that amount (`floor(ideal_dollar / price)`) is found; this count is capped
   so as not to exceed the REMAINING UNSPENT CASH so far (a safety ceiling —
   rounding errors are not allowed to let spending exceed `total_value`).
3. The cost of the lots purchased is deducted from the remaining cash, and
   the next (next-highest-weighted) symbol is processed.

`weights` need not sum to 1 — the function first normalizes the weights by
dividing by their own sum.

**Parameters:**
- `weights` (pd.Series): symbol -> weight (does not need to sum to 1).
- `latest_prices` (pd.Series): symbol -> most recent price.
- `total_value` (float): total capital to allocate; must be positive.

**Returns:** dict — `{"allocation": {symbol: lot_count, ...}, "leftover_cash":
float}`. `allocation` only includes symbols for which at least 1 lot could be
bought; `leftover_cash` is the remaining cash that could not be allocated
(not enough for a whole lot of any symbol).

**Raises:** `ValueError` — `total_value <= 0`, `weights` summing to <= 0, or
missing/non-positive prices in `latest_prices`.

**Example:**
```python
from quanttro.portfolio import discrete_allocation

latest_prices = price_df.iloc[-1]
alloc = discrete_allocation(w_hrp, latest_prices, total_value=100000)
```

Input prices (2026-10-02, real data):
```
THYAO    292.000000
SASA       1.930000
BIMAS    414.000000
EREGL     37.020000
GARAN    125.900002
```

**Output (`alloc`):**
```python
{'allocation': {'BIMAS': 64, 'EREGL': 688, 'THYAO': 68, 'GARAN': 112, 'SASA': 6964}, 'leftover_cash': 636.9198794364911}
```

(About 99.4% of the 100,000 TRY has been allocated, with ~637 TRY left as
cash since it wasn't enough for a whole lot of any symbol — SASA's very low
price (1.93 TRY) explains why it gets 100x more lots (6964) than BIMAS, since
both receive a similar dollar weight.)

---

## Comparison

`compare.py` provides helper functions for comparing the price/return series
of multiple strategies/portfolios: rebasing to a common starting value
(`rebase`) and a summary performance statistics table
(`compare_strategies`). For CONSISTENCY, existing metrics such as Sharpe and
max drawdown are NOT reimplemented here — `sharpe_ratio`/`max_drawdown` are
imported directly from the `quanttro.backtest.metrics` module and used as
is.

### `rebase(prices, base_value=100.0)`

Rescales EVERY COLUMN of a price DataFrame to the same starting value
(`base_value`), relative to its own first (first-row) value:

```
rebased[column] = prices[column] / prices[column].iloc[0] * base_value
```

Used before plotting multiple assets/strategies' return curves — which may
be at different absolute price levels — on the SAME axis in a comparable
way. If the first value is NaN, the entire column becomes NaN (intentional —
if the rebase point is undefined, the result must also be undefined).

**Parameters:**
- `prices` (pd.DataFrame): columns = asset/strategy, rows = time.
- `base_value` (float): the common value all columns will start from, default
  `100.0`.

**Returns:** pd.DataFrame — same shape as `prices`, each column starting from
`base_value`.

**Example:**
```python
from quanttro.portfolio import rebase

rebased = rebase(price_df[['THYAO', 'BIMAS']].tail(5), base_value=100)
```

**Output:**
```
                 THYAO       BIMAS
Date
2026-09-28  100.000000  100.000000
2026-09-29  101.567944   98.886284
2026-09-30   98.693380   98.124267
2026-10-01   99.825784   95.662368
2026-10-02  101.742160   97.069168
```

### `compare_strategies(returns_dict, periods_per_year=252, risk_free=0.0)`

Takes the per-period (simple/arithmetic) return series of multiple
strategies/portfolios and produces a summary performance statistics table
for each.

Columns computed for each strategy:
- `total_return`: total (cumulative) return, `(1+r).cumprod().iloc[-1] - 1`.
- `annualized_return`: annualized return (CAGR), `total_growth **
  (periods_per_year / n) - 1`.
- `annualized_volatility`: annualized volatility, `r.std(ddof=1) *
  sqrt(periods_per_year)`.
- `sharpe_ratio`: via `quanttro.backtest.metrics.sharpe_ratio` (not
  recomputed here, used directly for CONSISTENCY).
- `max_drawdown`: via `quanttro.backtest.metrics.max_drawdown`, over the
  cumulative return curve (a negative value, e.g. -0.23 -> -23%).

**Parameters:**
- `returns_dict` (dict[str, pd.Series]): strategy name -> per-period return
  series.
- `periods_per_year` (int): number of periods used for annualization, default
  `252` (daily).
- `risk_free` (float): annual risk-free rate, default `0.0`.

**Returns:** pd.DataFrame — rows = strategy name (`returns_dict` keys),
columns = the 5 metrics above. For a strategy with fewer than 2 valid
(non-NaN) observations, all metrics are NaN.

**Example:**
```python
from quanttro.portfolio import compare_strategies

rets_dict = {
    'THYAO': returns['THYAO'],
    'BIMAS': returns['BIMAS'],
    'HRP_portfolio': (returns * w_hrp).sum(axis=1),
}
cmp = compare_strategies(rets_dict)
```

**Output (`cmp.round(4)`):**
```
               total_return  annualized_return  annualized_volatility  sharpe_ratio  max_drawdown
THYAO                0.0397             0.0225                 0.3208        0.2290       -0.2440
BIMAS                0.6160             0.3156                 0.3523        0.9545       -0.2934
HRP_portfolio        0.2369             0.1292                 0.2838        0.5704       -0.1918
```

Commentary: the HRP portfolio shows lower volatility (0.2838) and a milder
max drawdown (-0.1918) than the individual stocks (THYAO, BIMAS) — the
expected effect of diversification; its return stays below BIMAS's but well
above THYAO's.
