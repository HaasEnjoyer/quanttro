# `quanttro.robustness`

This module collects four academically-defined robustness tools that are usually
missing from mainstream open-source backtesting libraries (vectorbt, backtrader, zipline):

- **`cpcv`** — Combinatorial Purged Cross-Validation (Lopez de Prado, 2018)
- **`overfitting`** — Probability of Backtest Overfitting / CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2017)
- **`montecarlo`** — Monte Carlo resampling over a return series
- **`sensitivity`** — one-at-a-time parameter sensitivity analysis

All of them are also accessible via `import quanttro as bq` (e.g. `bq.CombinatorialPurgedCV`,
`bq.probability_of_backtest_overfitting`, `bq.MonteCarloSimulator`, `bq.parameter_sensitivity`,
`bq.classify_sensitivity`), or directly from the `quanttro.robustness` subpackage.

The common thread running through all four tools: a "good" backtest result cannot be
judged from **a single number** or **a single path**. CPCV produces many different
paths instead of one train/test split; PBO measures, across those paths, whether the
strategy chosen as "best" is genuinely good or was picked by chance; Monte Carlo shows
how "typical" the one realized return path actually is; and sensitivity analysis tests
whether the "best" parameter is robust or fragile in its own neighborhood.

---

## 1. Combinatorial Purged Cross-Validation (CPCV)

**File:** `quanttro/robustness/cpcv.py` — **Reference:** Marcos López de Prado,
*Advances in Financial Machine Learning* (2018), Chapter 12.

### The problem: why standard k-fold CV is misleading for financial time series

Standard k-fold cross-validation rests on two basic assumptions: samples are
independent of each other, and the order of the train/test split doesn't matter. Both
assumptions break down for financial time series:

1. **Leakage.** A sample's label typically looks at a future window — e.g. "forward
   return over N bars" — and a feature typically looks at a past window — e.g. "rolling
   average of the last N bars." If the train and test sets are adjacent in time, these
   windows can cross the train/test boundary and leak into each other — the model
   effectively "sees" test-set information indirectly during training.
2. **A single train/test path.** Standard (even purged) k-fold CV produces only one
   train/test split. This makes it impossible to measure the variance of the
   performance estimate — and therefore the risk of "backtest overfitting": if the
   result looks good, there is no way to tell from a single path whether that is luck
   or a genuine signal.

### CPCV's solution

CPCV splits the data into `n_groups` equal parts and tries **every** combination of
using `n_test_groups` of them as the test set and the rest as the training set —
`C(n_groups, n_test_groups)` combinations in total. For `n_groups=6, n_test_groups=2`
this yields `C(6,2) = 15` distinct (and partially overlapping) train/test paths — each
one behaves like a different "alternate backtest history" of the same data.

**Important detail:** in CPCV the test groups do NOT have to be contiguous in time —
for example, group 0 and group 3 can be the test set while 1-2-4-5 are the training
set. That's why purge/embargo is applied not around the single block formed by the test
groups, but **separately around the boundaries of each individual test group**.

### Why purge/embargo is needed

- **`purge_gap`**: the number of contiguous (position-based) samples removed from
  training at the start and end of each test group. It prevents the label/feature
  windows of training samples from crossing the test boundary and causing leakage.
- **`embargo_pct`**: the size, as a fraction of the total data length, of the
  additional samples removed from training starting right after the END of each test
  group. It is applied only to samples that come after the test set (forward in time)
  — an extra safety margin because, due to serial correlation, post-test training
  samples can still be related to the test period. It is converted to an actual number
  of samples via `round(len(index) * embargo_pct)`.

These paths can also feed into the PBO computation in the
`quanttro.robustness.overfitting` module — the out-of-sample performance obtained
from each path can form the rows of the "multiple trials" matrix that PBO needs.

### `CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=0, embargo_pct=0.0)`

A Combinatorial Purged Cross-Validation split generator.

**Parameters:**
- `n_groups` (int): The number of equal parts (groups) to split the data into. Must be at least 2. Default `6`.
- `n_test_groups` (int): The number of groups used as the test set in each combination (`1 <= n_test_groups < n_groups`). Default `2`.
- `purge_gap` (int): The purge width described above (number of periods). Default `0`.
- `embargo_pct` (float): The embargo fraction described above, in `[0.0, 1.0)`. Default `0.0`.

Raises `ValueError` for invalid parameters (`n_groups < 2`, `n_test_groups` out of
range, `purge_gap < 0`, `embargo_pct` out of range).

#### `n_splits()`

Returns the total number of combinations: `math.comb(n_groups, n_test_groups)`.

**Returns:** `int`

#### `split(index)`

Generates all CPCV combinations over `index`.

**Parameters:**
- `index` (`pd.DatetimeIndex`): The time index of the data to be split.

**Returns:** `Iterator[tuple[np.ndarray, np.ndarray]]` — for each combination,
`(train_positions, test_positions)`. These are **positions within `index`, not
timestamps** (integer, ordered), so they can be used directly with
`df.iloc[train_positions]` / `df.iloc[test_positions]`. Combinations are generated in
`itertools.combinations` order (lexicographic).

**Example:**
```python
import pandas as pd
from quanttro.robustness.cpcv import CombinatorialPurgedCV

idx = pd.date_range("2020-01-01", periods=24, freq="D")
cpcv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=1, embargo_pct=0.05)
print("n_splits:", cpcv.n_splits())

splits = list(cpcv.split(idx))
train0, test0 = splits[0]
print("Split 0 - train positions:", train0)
print("Split 0 - test positions:", test0)
```

**Output:**
```
n_splits: 15
Split 0 - train positions: [10 11 12 13 14 15 16 17 18 19 20 21 22 23]
Split 0 - test positions: [0 1 2 3 4 5 6 7]
```

24 days of data are split into 6 groups (4 days each). In the first combination, group
0 and group 1 (days 0-7) are the test set; groups 2-5 (days 10-23) remain as the
training set. Because `purge_gap=1` and `embargo_pct=0.05` (`round(24*0.05)=1`), days 8
and 9 — starting right after the test block ends on day 7 (`purge_gap + embargo_size =
1 + 1 = 2` days) — are removed from the training set, so training jumps straight from
day 7 to day 10.

---

## 2. Probability of Backtest Overfitting (PBO)

**File:** `quanttro/robustness/overfitting.py` — **Reference:** Bailey, D. H., Borwein,
J., Lopez de Prado, M., & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting."
*Journal of Computational Finance.* The method is based on the **CSCV**
(Combinatorially Symmetric Cross-Validation) procedure.

### The problem

If a researcher tried N different strategy/parameter combinations and then selected
and reported the one that looks "best" (highest in-sample performance), that selection
itself is a **multiple testing** problem: the selected strategy is quite likely to have
come not from genuine skill, but from pure chance (the best-looking side of noise). PBO
estimates "the probability that the strategy selected as best in-sample fails to
perform well out-of-sample."

### Steps of the CSCV procedure

1. The T-periods × N-strategies performance matrix is split along the time axis into
   `S` (`n_partitions`) equal parts.
2. **All** `C(S, S/2)` combinations that can be formed from half (S/2) of these S parts
   are tried; in each combination, the chosen S/2 parts become IS (in-sample) and the
   remaining S/2 become OOS (out-of-sample).
3. For each combination: the strategy with the best IS performance is found (`n*`);
   this strategy's relative rank in the OOS data (rank among the N strategies,
   normalized to `(0,1)`) is computed.
4. This relative rank is passed through a **logit** transform (`ln(rank / (1 -
   rank))`). If the logit is `<= 0`, strategy `n*` ended up at or below the median
   OOS — i.e. its in-sample superiority was **not preserved** out-of-sample (a sign of
   overfitting).
5. **PBO** = the fraction of combinations where the logit is `<= 0`.

### Why PBO comes out low or high

For random trials (with no genuine signal), the strategy selected as "best" IS can end
up at any rank OOS purely by chance, so PBO is expected to come out **close to 0.5
(high)** — the probability that IS superiority is preserved OOS is as random as a coin
flip. A strategy with a consistent, genuine signal, on the other hand, will perform
systematically well both IS and OOS, so PBO comes out **low**.

### `probability_of_backtest_overfitting(trial_results, n_partitions=10)`

Computes PBO following the CSCV procedure.

**Parameters:**
- `trial_results` (`pd.DataFrame`): Rows = time periods (T), columns = different
  strategy/parameter trials (N, must be >= 2), values = return/performance in that
  period. Must not contain missing (NaN) values.
- `n_partitions` (int): The number of parts (`S`) the time axis is split into; must be
  **even** and at least 2 (so IS/OOS can each get S/2 equal parts). Default `10` →
  `C(10,5) = 252` combinations (keeps computational cost reasonable). The paper's
  default of S=16 (`C(16,8)=12870`) is more precise but more expensive.

**Returns:** `dict` — `{"pbo": float, "n_combinations_tested": int, "logits": list[float]}`
- `pbo`: in `[0, 1]`, the fraction of combinations with logit `<= 0`.
- `n_combinations_tested`: the number of combinations tested, `C(n_partitions, n_partitions/2)`.
- `logits`: the logit value computed for each combination (for distribution inspection/diagnostics).

Raises `ValueError` for invalid inputs (`n_partitions` odd or < 2, fewer than 2 columns,
fewer than `n_partitions` rows, data containing NaN).

Internal implementation note: a simple mean/std measure (an unannualized Sharpe-like
`mean/std`) is used as the performance score — in CSCV only the **relative** ranking
across strategies matters, so annualization is unnecessary.

### Proof: no signal vs. a real signal — a comparative example

The most important test proving this module's value: in a trial matrix that is
**entirely random** (no strategy systematically superior to any other), PBO should come
out high, while in a trial matrix with a **real, consistent signal**, PBO should come
out low.

```python
import numpy as np
import pandas as pd
from quanttro.robustness.overfitting import probability_of_backtest_overfitting

rng = np.random.default_rng(42)

# Scenario 1: NO SIGNAL - 20 "strategies", all drawn from the same (zero-mean, random) distribution
n_periods, n_trials = 500, 20
no_signal = pd.DataFrame(
    rng.normal(loc=0.0, scale=0.01, size=(n_periods, n_trials)),
    columns=[f"trial_{i}" for i in range(n_trials)],
)
result_no_signal = probability_of_backtest_overfitting(no_signal, n_partitions=10)
print("PBO (sinyal yok):", result_no_signal["pbo"])

# Scenario 2: REAL SIGNAL - trial_0 is consistently better than the others
real_signal = pd.DataFrame(
    rng.normal(loc=0.0, scale=0.01, size=(n_periods, n_trials)),
    columns=[f"trial_{i}" for i in range(n_trials)],
)
real_signal["trial_0"] = rng.normal(loc=0.0015, scale=0.01, size=n_periods)  # consistent positive drift
result_real_signal = probability_of_backtest_overfitting(real_signal, n_partitions=10)
print("PBO (gercek sinyal var):", result_real_signal["pbo"])
```

**Output:**
```
PBO (sinyal yok): 0.6547619047619048
PBO (gercek sinyal var): 0.11904761904761904
```

Both scenarios test `C(10,5) = 252` combinations with the same `n_partitions=10`. With
no signal, PBO comes out at **0.65** (a value at/above the median — the "best" IS
strategy generally doesn't rank well OOS, a strong sign of overfitting); once a
constant `+0.0015` drift is added to `trial_0`'s daily return, making it consistently
superior to the other 19 trials, PBO drops to **0.12** — i.e. the strategy selected as
best IS (in all likelihood always `trial_0`) stays above the OOS median in about 88% of
the combinations. This difference is direct evidence that PBO can distinguish a
strategy that is "best by chance" from one that is "genuinely best."

---

## 3. Monte Carlo Simulation

**File:** `quanttro/robustness/montecarlo.py`

The fact that a backtest relies on a single realized return path doesn't tell you how
much of a fluke that result might be. This module generates multiple alternative paths
(resampling) from the observed return series and is used to place the actual outcome
within that distribution.

### `MonteCarloSimulator(method="block_bootstrap", block_size=20, random_state=None)`

Monte Carlo resampling over a return series.

**Parameters:**
- `method` (str): `"block_bootstrap"` (default), `"shuffle"`, or `"iid_bootstrap"`.
- `block_size` (int): Only for `method="block_bootstrap"`: the length of each block (number of periods). Default `20`.
- `random_state` (`int | None`): Random seed for reproducibility (`None` = a different result on every call).

Raises `ValueError` for an invalid `method` or `block_size < 1`.

**Differences between the three methods:**
- **`"shuffle"`**: randomly shuffles the ORDER of the returns (tests the effect of
  trade order — would the result change if the same returns arrived in a different
  order?). DESTROYS the serial correlation (autocorrelation) between returns.
- **`"iid_bootstrap"`**: draws a sample from the returns, with replacement, under the
  assumption that they are independent and identically distributed. Like `"shuffle"`,
  it destroys serial correlation, and on top of that it reuses some returns and omits
  others entirely.
- **`"block_bootstrap"`**: consecutive blocks of length `block_size` (ordered
  sub-sequences from the original series) are randomly picked and concatenated until
  the original length is reached. Serial correlation within a block (e.g. volatility
  clustering, dependence between consecutive returns) **is preserved** — which is why
  it produces a more realistic alternative scenario than `"shuffle"`/`"iid_bootstrap"`.
  Blocks are chosen circularly (wrapping back to the start) when the end of the
  original series is reached.

#### `run(returns, n_sims=1000)`

Generates `n_sims` simulated cumulative return (equity) curves.

**Parameters:**
- `returns` (`pd.Series`): The realized return series (must contain at least 1 valid/non-NaN value).
- `n_sims` (int): The number of simulations to generate (at least 1). Default `1000`.

**Returns:** `pd.DataFrame` — each column is one simulation (named `0..n_sims-1`); each
curve is the same length as the original `returns` and is built as the multiplicative
cumulative product of `(1 + r)` (starting from 1.0). The row index is the same as
`returns.index` (can be used to visualize the time axis; in `"shuffle"`/`"iid_bootstrap"`
mode this is purely an axis label with no chronological meaning).

#### `summary(simulated_curves)`

Summarizes the simulation results.

**Parameters:**
- `simulated_curves` (`pd.DataFrame`): The output of `run()` (must contain at least one simulation/column).

**Returns:** `dict` —
`{"percentile_bands": pd.DataFrame, "max_drawdown_mean": float, "max_drawdown_median": float, "max_drawdown_p95_worst_case": float}`
- `percentile_bands`: a DataFrame containing, at each point in time, the 5/25/50/75/95
  percentile values across all simulations (suitable for fan-chart plotting; columns =
  `p5, p25, p50, p75, p95`).
- `max_drawdown_mean` / `max_drawdown_median`: the mean/median, across simulations, of
  each simulation's own maximum drawdown (a negative value, e.g. `-0.23`).
- `max_drawdown_p95_worst_case`: the value corresponding to the worst 5% tail of the
  drawdown distribution — since drawdown is negative, this is the (raw, signed) 5th
  percentile of the distribution (i.e. in 95% of simulations the maximum drawdown is
  milder/less negative than this value — the standard "worst case at the 95% confidence
  level" / VaR interpretation).

**Example (comparing all three methods):**
```python
import numpy as np
import pandas as pd
from quanttro.robustness.montecarlo import MonteCarloSimulator

rng = np.random.default_rng(7)
idx = pd.date_range("2022-01-01", periods=252, freq="B")
returns = pd.Series(rng.normal(0.0005, 0.012, size=len(idx)), index=idx)

for method in ["block_bootstrap", "shuffle", "iid_bootstrap"]:
    sim = MonteCarloSimulator(method=method, block_size=20, random_state=123)
    curves = sim.run(returns, n_sims=500)
    summary = sim.summary(curves)
    print(method, "->",
          "mean_dd:", round(summary["max_drawdown_mean"], 4),
          "median_dd:", round(summary["max_drawdown_median"], 4),
          "p95_worst_dd:", round(summary["max_drawdown_p95_worst_case"], 4))
```

**Output:**
```
block_bootstrap -> mean_dd: -0.3607 median_dd: -0.3606 p95_worst_dd: -0.5262
shuffle -> mean_dd: -0.3681 median_dd: -0.3647 p95_worst_dd: -0.4124
iid_bootstrap -> mean_dd: -0.3715 median_dd: -0.3723 p95_worst_dd: -0.5216
```

Notice that, with the same `random_state`, all three methods produce a similar mean
drawdown over the same `returns` series (this series doesn't have strong
autocorrelation in this example), but tail behavior (`p95_worst_case`) can differ by
method — the `percentile_bands` output (not shown in the example; it's a DataFrame with
the same number of rows as `curves.shape == (252, 500)` and columns `p5..p95`) can also
be used directly for fan-chart plotting.

---

## 4. Parameter Sensitivity

**File:** `quanttro/robustness/sensitivity.py`

It isn't enough for a strategy's "best" parameter combination to perform well at a
single point (a point estimate) — performance must also remain stable in the
neighborhood of that point (small changes in the parameter) (a **"plateau"**),
otherwise the result is fragile (a **"cliff"**) and is likely a product of overfitting.
This module scans each parameter one at a time (holding the others fixed — "one-at-a-time"
sensitivity analysis) and classifies whether the performance curve is a plateau or a
cliff.

### `parameter_sensitivity(backtest_func, param_grid, base_params)`

Scans each parameter, one at a time, over the values in `param_grid`, while holding the
others fixed at `base_params`.

**Parameters:**
- `backtest_func` (`Callable[..., float]`): A function with the signature
  `backtest_func(**params) -> float`; expected to return a single performance score
  (e.g. a Sharpe ratio).
- `param_grid` (`dict[str, list]`): A dict of `{parameter_name: [value1, value2, ...]}`
  listing the parameter values to scan.
- `base_params` (`dict`): A dict holding the fixed values for the parameters not being
  scanned (the scanned parameter's own entry in this dict is overridden at each step).

**Returns:** `pd.DataFrame` — a long-format table with one column (`"score"`), whose
rows are indexed by `(param_name, param_value)` pairs (a MultiIndex, with level names
`"param_name"`/`"param_value"`).

### `classify_sensitivity(sensitivity_df, param_name, plateau_threshold=0.1)`

Classifies a parameter's performance curve as `"plateau"` (robust) or `"cliff"` (fragile).

**Parameters:**
- `sensitivity_df` (`pd.DataFrame`): In the same format as `parameter_sensitivity`'s
  output (MultiIndex `param_name`/`param_value`, column `"score"`).
- `param_name` (str): The parameter to classify.
- `plateau_threshold` (float): The threshold value. Default `0.1`.

**Returns:** `str` — `"plateau"` or `"cliff"`.

The classification is based on the **coefficient of variation** (`|std / mean|`, using
the sample/`ddof=1` standard deviation) of that parameter's performance scores: if the
coefficient exceeds `plateau_threshold`, `"cliff"` is returned, otherwise `"plateau"`.
(Special case: if the mean is ~0, the ratio is undefined; in that case `"plateau"` is
returned if std is also ~0, otherwise `"cliff"`.)

Raises `ValueError` if `param_name` is not found in `sensitivity_df`, or if there are
fewer than 2 valid (non-NaN) scores for that parameter.

**Example (plateau vs. cliff comparison):**
```python
import numpy as np
from quanttro.robustness.sensitivity import parameter_sensitivity, classify_sensitivity

def fake_backtest(lookback: int, threshold: float) -> float:
    # "plateau" behavior: a smooth, wide performance plateau with respect to lookback
    plateau_score = 1.0 - 0.0001 * (lookback - 50) ** 2
    # "cliff" behavior: threshold causes a sharp performance drop past a certain point
    cliff_score = 1.2 if threshold < 0.05 else 0.1
    return plateau_score + cliff_score

param_grid = {
    "lookback": [20, 35, 50, 65, 80],
    "threshold": [0.01, 0.03, 0.05, 0.07, 0.09],
}
base_params = {"lookback": 50, "threshold": 0.03}

sens_df = parameter_sensitivity(fake_backtest, param_grid, base_params)
print(sens_df)
print("lookback siniflandirmasi:", classify_sensitivity(sens_df, "lookback"))
print("threshold siniflandirmasi:", classify_sensitivity(sens_df, "threshold"))
```

**Output:**
```
                         score
param_name param_value        
lookback   20.00        2.1100
           35.00        2.1775
           50.00        2.2000
           65.00        2.1775
           80.00        2.1100
threshold  0.01         2.2000
           0.03         2.2000
           0.05         1.1000
           0.07         1.1000
           0.09         1.1000

lookback siniflandirmasi: plateau
threshold siniflandirmasi: cliff
```

In the `lookback` scan, scores trace a smooth curve between 2.11 and 2.20 (coefficient
of variation ~2%, well below the threshold) → `"plateau"`: the strategy is robust to
this parameter. In the `threshold` scan, however, the score suddenly drops from 2.20 to
1.10 going from `0.03` to `0.05` (coefficient of variation ~39%, well above the
threshold) → `"cliff"`: the "best" value of this parameter (`0.03` or below) halves
performance under a small shift, so this parameter's selection is suspect for
overfitting and should not be trusted in live trading.
