# `quanttro.backtest`

This module provides four main components for model training and strategy evaluation:

1. **Walk-Forward Validation** (`WalkForwardSplitter`, `Fold`) - produces train/test splits on time series data without leaking future information (no look-ahead bias).
2. **Model Factory** (`ModelFactory`, `make_model`) - builds LightGBM, XGBoost, Random Forest, and linear (Ridge/LogisticRegression) models through a single interface, with conservative default hyperparameters.
3. **Risk/Performance Metrics** - 6 core metrics such as Sharpe, Sortino, Max Drawdown, Calmar, and the Deflated Sharpe Ratio; 14 extended metrics such as alpha/beta, Omega, VaR/CVaR, the Kelly criterion, and the Ulcer Index (20 metric functions in total).
4. **Cross-Sectional Ranking Evaluation** (`information_coefficient`, `tertile_spread`, `rolling_ic`) - measures a model's/signal's ranking performance across stocks (Information Coefficient, long-short spread).

All functions are accessible either at the top level (`import quanttro as bq`) or via the submodule (`from quanttro.backtest import metrics, ranking, walkforward, models`). All the examples below were actually executed, and their outputs were taken directly from the console; the data used is the `ret_1` (daily simple return) column from the files `data/features/daily/THYAO.csv` (and, for benchmarking, `BIMAS.csv`, `AKBNK.csv`, `ARCLK.csv`, `ASELS.csv`), over a 935-day subset after 2023-01-01 (`r_recent`).

Summary statistics for the sample data:

```python
import pandas as pd
df = pd.read_csv("data/features/daily/THYAO.csv", parse_dates=["Date"]).set_index("Date")
r_recent = df["ret_1"].dropna().loc["2023-01-01":]
print(r_recent.describe())
```

**Output:**
```
count    935.000000
mean       0.000988
std        0.023601
min       -0.095912
25%       -0.012913
50%       -0.000769
75%        0.012797
max        0.120000
Name: ret_1, dtype: float64
```

---

## 1. Walk-Forward Validation

### `Fold`

A `@dataclass(frozen=True)` holding the boundaries of a single walk-forward fold. For backward compatibility, it can be unpacked/indexed like a 3-element tuple of `(train_cutoff, test_start, test_end)` (`__iter__`, `__getitem__`, `__len__` are defined). Since the rolling-window mode may also need the LOWER bound of the training set, this is tracked separately in the `train_start` field.

**Fields:**
- `train_cutoff` (`pd.Timestamp`): the end date (inclusive) of the training set.
- `test_start` (`pd.Timestamp`): the first date of the test block.
- `test_end` (`pd.Timestamp`): the last date of the test block.
- `train_start` (`pd.Timestamp`): the start date of the training set. In expanding mode, this is the very first date in the data; in rolling mode, it's the LOWER bound as the training window slides forward - in rolling mode, using only `train_cutoff` (the old expanding-style filtering) would mistakenly reintroduce expanding behavior, so the `index >= train_start` filter must also be applied in rolling mode.

**Example:**
```python
fold = next(iter(bq.WalkForwardSplitter(n_folds=3).split(df.index)))
print(fold)
print(tuple(fold))
print(len(fold))
```

**Output:**
```
Fold(train_cutoff=Timestamp('2010-08-17 00:00:00'), test_start=Timestamp('2010-08-18 00:00:00'), test_end=Timestamp('2015-11-27 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
(Timestamp('2010-08-17 00:00:00'), Timestamp('2010-08-18 00:00:00'), Timestamp('2015-11-27 00:00:00'))
3
```

---

### `WalkForwardSplitter(n_folds=5, purge_gap=0, window="expanding", train_window=None)`

Implements walk-forward (forward-moving) splitting logic for time series/panel data. The data is split into `n_folds + 1` equal parts over unique dates (using `numpy.array_split`); the first part is never tested and only feeds into the first fold's training pool, while the remaining `n_folds` parts are used sequentially as test blocks.

**Parameters:**
- `n_folds` (int): the number of folds to produce, default `5`. Cannot be less than 1 (`ValueError`).
- `purge_gap` (int): the number of unique dates to leave as a gap between train and test, removed from training, default `0` (no gap). Cannot be negative (`ValueError`). Used to reduce information leakage at the train/test boundary for forward-looking (N-bar-ahead) target variables (a simplified version of Lopez de Prado's "purged cross-validation" idea).
- `window` (str): `"expanding"` (default, a growing window - the training set always starts from the beginning of the data) or `"rolling"` (a fixed-width sliding window). Any other value raises `ValueError`.
- `train_window` (int | None): only used with `window="rolling"`, the width of the training window in number of blocks. If `None`, it is treated as `1`. Cannot be less than `1` (`ValueError`).

**Methods:**
- `expanding_window(n_folds=5, purge_gap=0)` (classmethod): returns a `WalkForwardSplitter` in expanding-window mode.
- `rolling_window(n_folds=5, purge_gap=0, train_window=1)` (classmethod): returns a `WalkForwardSplitter` in rolling-window mode.
- `split(index: pd.DatetimeIndex) -> Iterator[Fold]`: generates folds over the given index (typically a panel index with repeated dates). Raises `ValueError` if the number of unique dates is less than `n_folds + 1`.

**Returns (`split`):** `Iterator[Fold]`

**Example (default expanding, n_folds=3):**
```python
import quanttro as bq

splitter = bq.WalkForwardSplitter(n_folds=3)
folds = list(splitter.split(df.index))
print("n fold:", len(folds))
for f in folds:
    print(f)
```

**Output:**
```
n fold: 3
Fold(train_cutoff=Timestamp('2010-08-17 00:00:00'), test_start=Timestamp('2010-08-18 00:00:00'), test_end=Timestamp('2015-11-27 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
Fold(train_cutoff=Timestamp('2015-11-27 00:00:00'), test_start=Timestamp('2015-11-30 00:00:00'), test_end=Timestamp('2021-03-25 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
Fold(train_cutoff=Timestamp('2021-03-25 00:00:00'), test_start=Timestamp('2021-03-26 00:00:00'), test_end=Timestamp('2026-09-25 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
```

In every fold, `train_start` stays fixed (`2005-03-28`, the very beginning of the data) - this is the proof of expanding behavior: the training set keeps growing.

**Example (`expanding_window` with `purge_gap=5`):**
```python
from quanttro.backtest import walkforward

sp2 = walkforward.WalkForwardSplitter.expanding_window(n_folds=2, purge_gap=5)
for f in sp2.split(df.index):
    print(f)
```

**Output:**
```
Fold(train_cutoff=Timestamp('2012-05-14 00:00:00'), test_start=Timestamp('2012-05-22 00:00:00'), test_end=Timestamp('2019-06-05 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
Fold(train_cutoff=Timestamp('2019-05-29 00:00:00'), test_start=Timestamp('2019-06-06 00:00:00'), test_end=Timestamp('2026-09-25 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
```

Thanks to `purge_gap=5`, you can see a 5-day gap forming between `train_cutoff` and `test_start` (e.g. in the first fold, `train_cutoff=2012-05-14`, `test_start=2012-05-22` - the days in between have been excluded from training).

**Example (`rolling_window` with `train_window=2`):**
```python
sp3 = walkforward.WalkForwardSplitter.rolling_window(n_folds=4, train_window=2)
for f in sp3.split(df.index):
    print(f)
```

**Output:**
```
Fold(train_cutoff=Timestamp('2009-07-20 00:00:00'), test_start=Timestamp('2009-07-21 00:00:00'), test_end=Timestamp('2013-10-16 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
Fold(train_cutoff=Timestamp('2013-10-16 00:00:00'), test_start=Timestamp('2013-10-17 00:00:00'), test_end=Timestamp('2018-01-05 00:00:00'), train_start=Timestamp('2005-03-28 00:00:00'))
Fold(train_cutoff=Timestamp('2018-01-05 00:00:00'), test_start=Timestamp('2018-01-08 00:00:00'), test_end=Timestamp('2022-04-29 00:00:00'), train_start=Timestamp('2009-07-21 00:00:00'))
Fold(train_cutoff=Timestamp('2022-04-29 00:00:00'), test_start=Timestamp('2022-05-05 00:00:00'), test_end=Timestamp('2026-09-25 00:00:00'), train_start=Timestamp('2013-10-17 00:00:00'))
```

In rolling mode, `train_start` stays fixed for the first two folds (until 2 blocks have accumulated), and starting from the 3rd fold it begins shifting forward to preserve the `train_window=2` block width (`train_start` goes `2005-03-28` -> `2009-07-21` -> `2013-10-17`) - this shows that the training window slides forward at a FIXED width.

---

## 2. Model Factory

### `ModelFactory`

A factory class for producing regression/classification models. Supported `model_type` values: `"lightgbm"`, `"xgboost"`, `"random_forest"`, `"linear"` (`Ridge` for regression, `LogisticRegression` for classification). The default hyperparameters follow a conservative philosophy that constrains tree depth/leaf count and keeps the minimum samples per leaf/node high (to limit overfitting risk when working with a small number of stocks/observations).

#### `ModelFactory.available_models() -> tuple[str, ...]`

Returns all supported `model_type` values as an alphabetically sorted tuple (`staticmethod`).

**Returns:** `tuple[str, ...]`

**Example:**
```python
print(bq.ModelFactory.available_models())
```

**Output:**
```
('lightgbm', 'linear', 'random_forest', 'xgboost')
```

#### `ModelFactory.create(model_type, task="regression", **overrides)`

Returns a model object (with a `fit`/`predict` interface) for the given `model_type` and `task`, with `overrides` applied on top of the default hyperparameters (`classmethod`).

**Parameters:**
- `model_type` (str): `"lightgbm"`, `"xgboost"`, `"random_forest"`, or `"linear"` (case-insensitive). An invalid value raises `ValueError`.
- `task` (str): `"regression"` or `"classification"`, default `"regression"`. An invalid value raises `ValueError`.
- `**overrides`: additional/modified `kwargs` to override each model type's default hyperparameters.

**Default hyperparameters by model type (from source code):**
- `"lightgbm"`: `n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1` -> `lgb.LGBMRegressor` for regression, `lgb.LGBMClassifier` for classification.
- `"xgboost"`: `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0, n_jobs=-1, verbosity=0` (plus an `eval_metric="logloss"` default for classification) -> `xgb.XGBRegressor` / `xgb.XGBClassifier`. Raises a clear `ImportError` message if the `xgboost` package is not installed.
- `"random_forest"`: `n_estimators=200, max_depth=4, min_samples_leaf=50, n_jobs=-1, random_state=42` -> `RandomForestRegressor` / `RandomForestClassifier`.
- `"linear"`: `Ridge(alpha=1.0)` for regression, `LogisticRegression(C=1.0, max_iter=1000)` for classification.

**Returns:** a model object with a `fit`/`predict` interface (e.g. `sklearn.ensemble.RandomForestRegressor`, `lightgbm.LGBMRegressor`, etc.)

**Example:**
```python
m = bq.ModelFactory.create("random_forest", task="regression", n_estimators=50)
print(m)
```

**Output:**
```
RandomForestRegressor(max_depth=4, min_samples_leaf=50, n_estimators=50,
                      n_jobs=-1, random_state=42)
```

You can see that the `n_estimators=50` override replaced the default `200`, while the other defaults (`max_depth=4`, `min_samples_leaf=50`, `random_state=42`) remained unchanged.

---

### `make_model(task, model_type="lightgbm", **overrides)`

A shortcut function backward-compatible with the original `make_model` in `train_model.py`: it produces LightGBM by default, but another model type can also be selected via `model_type`. Internally, it calls `ModelFactory.create(model_type=model_type, task=task, **overrides)`.

**Parameters:**
- `task` (str): `"regression"` or `"classification"`.
- `model_type` (str): default `"lightgbm"`.
- `**overrides`: hyperparameter overrides.

**Returns:** a model object with a `fit`/`predict` interface.

**Example:**
```python
m2 = bq.make_model(task="classification", model_type="linear")
print(m2)

m3 = bq.make_model(task="regression")  # default: lightgbm
print(m3)
```

**Output:**
```
LogisticRegression(max_iter=1000)
LGBMRegressor(learning_rate=0.05, max_depth=4, min_child_samples=50,
              n_estimators=200, num_leaves=15, verbose=-1)
```

---

## 3. Risk/Performance Metrics

Note: all functions operate on per-period (daily/hourly/etc.) **simple (arithmetic, not compounded)** return series. All the examples below were run on THYAO's daily return series after 2023-01-01 (`r_recent`, 935 observations) and the cumulative growth curve `curve = (1 + r_recent).cumprod()`.

### Core Metrics (6)

#### `sharpe_ratio(returns, periods_per_year=252, risk_free=0.0) -> float`

The annualized Sharpe ratio:

```
Sharpe = mean(r_excess) / std(r_excess, ddof=1) * sqrt(periods_per_year)
```

**Parameters:**
- `returns`: the return series (list/array/Series, NaNs are automatically dropped).
- `periods_per_year` (int): the annualization factor, default `252` (number of trading days).
- `risk_free` (float): the ANNUAL risk-free rate, default `0.0`. It is converted to a per-period rate (`risk_free / periods_per_year`) and subtracted from the returns.

**Returns:** `float`. `NaN` if there are fewer than 2 returns or the standard deviation is ~0.

**Example:**
```python
from quanttro.backtest import metrics
print(metrics.sharpe_ratio(r_recent))
```

**Output:**
```
0.6643588511095899
```

#### `sortino_ratio(returns, periods_per_year=252, risk_free=0.0) -> float`

Similar to Sharpe, but the denominator only uses downside deviation - upside volatility is not penalized:

```
downside_dev = sqrt( mean( min(r_excess, 0)^2 ) )   (over ALL periods)
Sortino = mean(r_excess) / downside_dev * sqrt(periods_per_year)
```

**Parameters:** same as `sharpe_ratio` (`returns`, `periods_per_year=252`, `risk_free=0.0`).

**Returns:** `float`. NaN conditions are the same as `sharpe_ratio`.

**Example / Output:**
```python
print(metrics.sortino_ratio(r_recent))
```
```
1.0431460862143038
```

#### `max_drawdown(cumulative_returns) -> float`

The worst percentage decline of a cumulative return/equity curve relative to its running peak up to that point:

```
drawdown_t = curve_t / running_max_t - 1
max_drawdown = min(drawdown_t)
```

**Parameters:**
- `cumulative_returns`: the cumulative growth curve (e.g. `(1 + r).cumprod()`), **not a return series**.

**Returns:** `float`, NEGATIVE (e.g. `-0.23` -> a 23% maximum decline). NaN for an empty series.

**Example / Output:**
```python
curve = (1 + r_recent).cumprod()
print(metrics.max_drawdown(curve))
```
```
-0.29432133525767046
```

#### `calmar_ratio(returns, periods_per_year=252) -> float`

The Calmar ratio = annualized return (CAGR) / |maximum drawdown|. The cumulative growth curve is built from the return series, and CAGR and max drawdown are computed from that curve.

**Parameters:**
- `returns`: the return series.
- `periods_per_year` (int): default `252`.

**Returns:** `float`. NaN if `n < 2`, `mdd == 0`/NaN, or total growth is `<= 0`.

**Example / Output:**
```python
print(metrics.calmar_ratio(r_recent))
```
```
0.6670901610217087
```

#### `hit_rate(returns) -> float`

The positive-return ratio: the fraction of periods with returns greater than 0 out of total periods (a return of exactly 0 does not count as positive).

**Parameters:** `returns`: the return series.

**Returns:** `float` (between 0 and 1). NaN for an empty series.

**Example / Output:**
```python
print(metrics.hit_rate(r_recent))
```
```
0.47593582887700536
```

#### `profit_factor(returns) -> float`

Profit factor = total positive returns / |total negative returns|.

**Parameters:** `returns`: the return series.

**Returns:** `float`. `+inf` if there are no losses and there are gains; NaN if there are neither gains nor losses.

**Example / Output:**
```python
print(metrics.profit_factor(r_recent))
```
```
1.12185395038729
```

#### `deflated_sharpe_ratio(sharpe, n_trials, skewness, kurtosis, n_observations) -> float`

The Deflated Sharpe Ratio (DSR) - Bailey & Lopez de Prado (2014). It corrects for the risk that testing multiple strategies/parameter combinations (`n_trials`) can produce a high Sharpe ratio by chance, and for how far the return distribution deviates from normality (`skewness`, `kurtosis`), yielding the probability that the "true Sharpe ratio > 0" on a `[0, 1]` scale.

Method:
1. The standard error of the Sharpe estimate: `se(SR) = sqrt( (1 - skew*SR + (kurtosis-1)/4*SR^2) / (n_observations - 1) )`
2. The expected maximum Sharpe threshold (`SR_0`) that `n_trials` attempts could produce by chance is computed using the Euler-Mascheroni constant and extreme value theory: `SR_0 = se(SR) * [ (1-gamma)*Z^-1(1-1/n_trials) + gamma*Z^-1(1-1/(n_trials*e)) ]`
3. `DSR = CDF( (SR - SR_0) / se(SR) )`

**Parameters:**
- `sharpe` (float): the Sharpe ratio being evaluated (per-period use is recommended; the function does not perform any conversion).
- `n_trials` (int): the number of strategy/parameter combinations tried. If `1`, no multiple-testing correction is applied (the comparison threshold is taken as 0, equivalent to the classic Probabilistic Sharpe Ratio). Cannot be less than `1` (`ValueError`).
- `skewness` (float): the skewness of the return series.
- `kurtosis` (float): the FULL (Pearson) kurtosis of the return series - `3` for a normal distribution. Should be computed with `scipy.stats.kurtosis(x, fisher=False)`; NOT pandas/scipy's default "excess" kurtosis (0 for a normal distribution).
- `n_observations` (int): the number of observations.

**Returns:** `float` (a probability between 0 and 1).

**Example:**
```python
from scipy.stats import skew, kurtosis as kurt

sk = skew(r_recent)
ku = kurt(r_recent, fisher=False)
sr_period = r_recent.mean() / r_recent.std(ddof=1)
print("period sharpe:", sr_period, "skew:", sk, "kurtosis(pearson):", ku)

print(metrics.deflated_sharpe_ratio(sharpe=sr_period, n_trials=50, skewness=sk, kurtosis=ku, n_observations=len(r_recent)))
print(metrics.deflated_sharpe_ratio(sharpe=sr_period, n_trials=1, skewness=sk, kurtosis=ku, n_observations=len(r_recent)))
```

**Output:**
```
period sharpe: 0.04185067384144197 skew: 0.5055281453455335 kurtosis(pearson): 5.429378036997248
0.16235679380186746
0.9017322784840389
```

When the trial-count correction is applied with `n_trials=50`, the DSR drops to `0.16` (meaning the "best of" 50 trials could by chance produce this much Sharpe), while with `n_trials=1` (no correction, classic PSR) the DSR comes out to `0.90` - showing how sensitive the same Sharpe ratio is to the number of trials.

---

### Extended Metrics (14)

For the outputs below, the `alpha_beta` and `capture_ratio` functions use THYAO (`r_recent`) and, as the benchmark series, BIMAS's return over the same period (`bimas_recent`).

#### `alpha_beta(returns, benchmark_returns, risk_free=0.0, periods_per_year=252) -> tuple[float, float]`

CAPM alpha (annualized) and beta. The return series are aligned over their common (index intersection) periods.

```
beta = Cov(r_excess, b_excess) / Var(b_excess)     (sample covariance/variance, ddof=1)
alpha_period = mean(r_excess) - beta * mean(b_excess)
alpha = alpha_period * periods_per_year
```

**Parameters:**
- `returns`, `benchmark_returns`: return series.
- `risk_free` (float): the annual risk-free rate, default `0.0`.
- `periods_per_year` (int): default `252`.

**Returns:** `tuple[float, float]` -> `(alpha, beta)`. `(nan, nan)` if there are fewer than 2 common observations or the benchmark's excess variance is ~0.

**Example / Output:**
```python
bimas_recent = pd.read_csv("data/features/daily/BIMAS.csv", parse_dates=["Date"]).set_index("Date")["ret_1"].dropna().loc["2023-01-01":]
print(metrics.alpha_beta(r_recent, bimas_recent))
```
```
(-0.02729063011028415, 0.47138988356224615)
```

#### `omega_ratio(returns, required_return=0.0) -> float`

The sum of gains above a threshold divided by the absolute sum of losses below the threshold: `Omega = sum(excess[excess>0]) / |sum(excess[excess<0])|` (`excess = r - required_return`).

**Parameters:**
- `returns`: the return series.
- `required_return` (float): the PER-PERIOD threshold return, default `0.0`.

**Returns:** `float`. `+inf` if there are no losses below the threshold and the numerator > 0; NaN if both are 0.

**Example / Output:**
```python
print(metrics.omega_ratio(r_recent))
```
```
1.12185395038729
```

(Since `required_return=0.0`, this is mathematically identical to `profit_factor` - the expected behavior.)

#### `tail_ratio(returns) -> float`

Tail ratio = 95th percentile / |5th percentile| (using `np.percentile` with linear interpolation).

**Parameters:** `returns`: the return series.

**Returns:** `float`. NaN if the 5th percentile is 0.

**Example / Output:**
```python
print(metrics.tail_ratio(r_recent))
```
```
1.2811639115516567
```

#### `value_at_risk(returns, cutoff=0.05) -> float`

Historical VaR, percentile method: `VaR = percentile(r, cutoff * 100)`.

**Parameters:**
- `returns`: the return series.
- `cutoff` (float): in the open interval (0,1), default `0.05`. A value outside this range raises `ValueError`.

**Returns:** `float`, typically NEGATIVE.

**Example / Output:**
```python
print(metrics.value_at_risk(r_recent))  # cutoff=0.05
```
```
-0.03283509363321615
```

#### `conditional_value_at_risk(returns, cutoff=0.05) -> float`

CVaR/Expected Shortfall: the average of returns AT OR BELOW the VaR threshold (threshold inclusive).

**Parameters:** same as `value_at_risk`.

**Returns:** `float`, typically NEGATIVE, with `|CVaR| >= |VaR|` in magnitude.

**Example / Output:**
```python
print(metrics.conditional_value_at_risk(r_recent))
```
```
-0.04783999129235774
```

#### `capture_ratio(returns, benchmark_returns, side="up") -> float`

The upside/downside capture ratio. `side="up"`: the strategy's average return / the benchmark's average return when the benchmark is positive; `side="down"`: the same ratio when the benchmark is negative.

**Parameters:**
- `returns`, `benchmark_returns`: return series.
- `side` (str): `"up"` or `"down"`, default `"up"`. Any other value raises `ValueError`.

**Returns:** `float`. NaN if the relevant subset of the benchmark is empty/has a mean of 0.

**Example / Output:**
```python
print(metrics.capture_ratio(r_recent, bimas_recent, side="up"))
print(metrics.capture_ratio(r_recent, bimas_recent, side="down"))
```
```
0.4458102780510679
0.4588851955552379
```

#### `avg_win(returns) -> float`

The average of positive returns.

**Returns:** `float`. NaN if there are no gains.

**Example / Output:**
```python
print(metrics.avg_win(r_recent))
```
```
0.019106693753970994
```

#### `avg_loss(returns) -> float`

The average of negative returns (a NEGATIVE value).

**Returns:** `float`. NaN if there are no losses.

**Example / Output:**
```python
print(metrics.avg_loss(r_recent))
```
```
-0.016125432593230276
```

#### `payoff_ratio(returns) -> float`

`avg_win / |avg_loss|`.

**Returns:** `float`. NaN if there are no losses or no gains.

**Example / Output:**
```python
print(metrics.payoff_ratio(r_recent))
```
```
1.1848794532180364
```

#### `kelly_criterion(returns) -> float`

The simple Kelly criterion (f*), an estimate of the optimal bet/leverage fraction that maximizes log growth:

```
win_rate = hit_rate(r)
win_loss_ratio = payoff_ratio(r)
f* = win_rate - (1 - win_rate) / win_loss_ratio
```

**Returns:** `float`. NaN if `win_loss_ratio` cannot be computed or is 0. A negative result indicates it would be better not to bet.

**Example / Output:**
```python
print(metrics.kelly_criterion(r_recent))
```
```
0.033642589932166844
```

#### `risk_of_ruin(returns, risk_fraction=1.0) -> float`

A simplified "risk of ruin" (probability of capital depletion) estimate - it uses ONLY the win rate (win_rate), not the magnitude of returns:

```
edge = 2 * win_rate - 1
units = 1 / risk_fraction
RoR = ((1 - edge) / (1 + edge)) ** units    (for edge > 0)
```
`RoR = 1.0` if `edge <= 0`; `RoR = 0.0` if `edge >= 1`.

**Parameters:**
- `returns`: the return series.
- `risk_fraction` (float): the fraction of capital risked per trade, default `1.0`. Must be greater than `0`.

**Returns:** `float` (between 0 and 1).

**Example / Output:**
```python
print(metrics.risk_of_ruin(r_recent, risk_fraction=0.02))
```
```
1.0
```

Since THYAO's `hit_rate` over this period (`0.476`) is below 50%, `edge <= 0`, so the function treats ruin as "practically certain" (`1.0`) - an expected result showing how sensitive this simple model is to the win rate.

#### `ulcer_index(cumulative_returns) -> float`

The RMS of the drawdown series - it penalizes both the depth and the duration of drawdowns: `UI = sqrt(mean(drawdown_t^2))`.

**Parameters:** `cumulative_returns`: the cumulative growth curve.

**Returns:** `float`, always `>= 0`.

**Example / Output:**
```python
print(metrics.ulcer_index(curve))
```
```
0.12360866670594178
```

#### `common_sense_ratio(returns) -> float`

`CSR = tail_ratio(r) * profit_factor(r)`.

**Returns:** `float`.

**Example / Output:**
```python
print(metrics.common_sense_ratio(r_recent))
```
```
1.4372787952678585
```

#### `recovery_factor(cumulative_returns) -> float`

`Recovery Factor = total_return / |max_drawdown|` (total return being the ratio of the cumulative curve's last value to its first).

**Returns:** `float`. Requires at least 2 observations; NaN if `mdd` is 0/NaN.

**Example / Output:**
```python
print(metrics.recovery_factor(curve))
```
```
2.9443642370309497
```

#### `gain_to_pain_ratio(returns) -> float`

`total_return (simple sum) / sum of |negative returns|`.

**Returns:** `float`. `+inf` if there are no losses and the numerator > 0; NaN if both are 0.

**Example / Output:**
```python
print(metrics.gain_to_pain_ratio(r_recent))
```
```
0.12185395038728988
```

---

## 4. Cross-Sectional Ranking Evaluation

The functions in this section measure a model's/signal's ranking performance ACROSS stocks (not the time series of a single stock, but the cross-section across multiple stocks on a given date). For the example, a panel (935x5) was built from the daily returns of 5 stocks (`THYAO, BIMAS, AKBNK, ARCLK, ASELS`) after 2023-01-01, using each stock's PRIOR day's return (`shift(1)`) as a simple "signal."

```python
tickers = ["THYAO", "BIMAS", "AKBNK", "ARCLK", "ASELS"]
panel = pd.DataFrame({t: pd.read_csv(f"data/features/daily/{t}.csv", parse_dates=["Date"]).set_index("Date")["ret_1"] for t in tickers}).dropna()
panel_recent = panel.loc["2023-01-01":]
```

### `information_coefficient(predicted, actual, method="spearman") -> float`

The Information Coefficient (IC) for a single period/cross-section: the correlation between the predicted ranking and the realized returns.

**Parameters:**
- `predicted` (`pd.Series`): predicted values (index = stock/symbol).
- `actual` (`pd.Series`): realized returns (same index space).
- `method` (str): `"spearman"` (default, rank correlation), `"pearson"`, or `"kendall"` (whichever `pandas.Series.corr` supports).

**Returns:** `float`. NaN if the number of valid (non-NaN, common-index) points is fewer than 3.

**Example:**
```python
from quanttro.backtest import ranking

pred_signal = panel_recent.shift(1).iloc[100]   # 2023-05-31, the prior day's return as the "signal"
actual_day = panel_recent.iloc[100]             # that same day's realized return
print(pred_signal)
print(actual_day)
print(ranking.information_coefficient(pred_signal, actual_day))
```

**Output:**
```
THYAO    0.006849
BIMAS    0.024781
AKBNK    0.099396
ARCLK    0.074335
ASELS    0.018711
Name: 2023-05-31 00:00:00, dtype: float64

THYAO   -0.021769
BIMAS    0.002845
AKBNK   -0.034820
ARCLK   -0.027273
ASELS   -0.030612
Name: 2023-05-31 00:00:00, dtype: float64

-0.49999999999999994
```

The IC came out negative (`-0.5`): the stock that rose the most the previous day (AKBNK) ended up among the biggest losers this day - showing that, in this particular small 5-stock cross-section, the "momentum" signal worked in reverse on this specific day (this is illustrative, not a general result).

### `tertile_spread(predicted, actual, n_groups=3) -> float`

The top-minus-bottom group long-short spread: the average realized return of the highest-predicted group, ranked by prediction, MINUS the average realized return of the lowest-predicted group.

**Parameters:**
- `predicted`, `actual` (`pd.Series`): as above.
- `n_groups` (int): the number of groups, default `3` (tertile). Quintile (`5`), etc. can also be used. Cannot be less than `2` (`ValueError`). Group size is `len(universe) // n_groups`, at minimum `1`.

**Returns:** `float`. NaN if the number of valid points is less than `n_groups`.

**Example / Output:**
```python
print(ranking.tertile_spread(pred_signal, actual_day, n_groups=3))
```
```
-0.0130511533799588
```

(With 5 stocks and `n_groups=3` -> group size `5 // 3 = 1`; the difference is taken between the realized return of the 1 highest-predicted stock and the 1 lowest-predicted stock.)

### `rolling_ic(predicted_by_date, actual_by_date, window=60, method="spearman") -> pd.Series`

Computes the cross-sectional IC at each date, then returns the `window`-period rolling average of that IC series.

**Parameters:**
- `predicted_by_date`, `actual_by_date` (`pd.DataFrame`): wide-format DataFrames with dates as rows and stocks/symbols as columns. Aligned on their inner intersection by index (date).
- `window` (int): the rolling average window, default `60`.
- `method` (str): `"spearman"` (default), `"pearson"`, or `"kendall"`.

If a date has fewer than 3 valid (common, non-NaN) symbols, that day's IC is treated as NaN. `min_periods = max(2, window // 3)`.

**Returns:** `pd.Series` (index = date, `name="ic"`).

**Example:**
```python
pred_by_date = panel_recent.shift(1).dropna()
actual_by_date = panel_recent.loc[pred_by_date.index]
ric = ranking.rolling_ic(pred_by_date, actual_by_date, window=20)
print(ric.tail())
print(ric.describe())
```

**Output:**
```
2026-09-21   -4.500000e-02
2026-09-22    2.220446e-17
2026-09-23   -1.000000e-02
2026-09-24   -5.000000e-03
2026-09-25   -1.000000e-02
Name: ic, dtype: float64

count    929.000000
mean       0.023805
std        0.108512
min       -0.270000
25%       -0.050000
50%        0.025000
75%        0.095000
max        0.375000
Name: ic, dtype: float64
```

The average cross-sectional IC (`0.024`) is slightly positive but quite weak - this shows that this simple "prior day's return" signal has no systematic predictive power in this small 5-stock universe (or it's very weak at best); the `std` (`0.109`) reveals substantial day-to-day fluctuation.

---

## Summary: `quanttro.backtest.__all__`

All public names exported by the module's `__init__.py`: `Fold`, `WalkForwardSplitter`, `ModelFactory`, `make_model`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `hit_rate`, `profit_factor`, `deflated_sharpe_ratio`, `information_coefficient`, `tertile_spread`, `rolling_ic`, `alpha_beta`, `omega_ratio`, `tail_ratio`, `value_at_risk`, `conditional_value_at_risk`, `capture_ratio`, `avg_win`, `avg_loss`, `payoff_ratio`, `kelly_criterion`, `risk_of_ruin`, `ulcer_index`, `common_sense_ratio`, `recovery_factor`, `gain_to_pain_ratio`. This document covers all of the names above (2 classes + 1 standalone function on the walk-forward/model side, in total, plus 20 metric functions, and 3 ranking functions).
