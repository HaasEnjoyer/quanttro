# Case study: is the "Golden Cross" overfit?

*A worked example of `quanttro.robustness` applied to a real, extremely
popular retail strategy. Every number below was actually computed — see
the code in each section; nothing is illustrative or hypothetical.*

> **Disclaimer:** This is a methodology demonstration, not a trading
> recommendation. See the project [disclaimer](../README.md) — nothing
> here is investment advice.

## The strategy

The "Golden Cross" — go long when a short moving average crosses above a
long moving average, flat otherwise — is one of the most widely cited
technical rules in retail trading, usually quoted as the 50-day/200-day
pair. The question this case study asks: **is 50/200 actually special, or
is it one lucky draw among many similar pairs that would have looked just
as good?**

## Setup

Real daily SPY data (1993–2026, 8,481 trading days) via `quanttro.fetch`.
16 moving-average pairs were tested (short ∈ {10, 20, 30, 50}, long ∈
{100, 150, 200, 250}, short < long always):

```python
import pandas as pd
import quanttro as qt

df = qt.fetch("SPY")
close = df["Close"]
ret_1 = close.pct_change()

trial_returns = {}
for s in [10, 20, 30, 50]:
    for l in [100, 150, 200, 250]:
        if s >= l:
            continue
        signal = (close.rolling(s).mean() > close.rolling(l).mean()).astype(float)
        trial_returns[f"MA{s}_{l}"] = signal.shift(1) * ret_1  # shift(1): no lookahead

trial_df = pd.DataFrame(trial_returns).dropna()
```

Each column is one "trial" — exactly the `trial_results` input
`quanttro.robustness.probability_of_backtest_overfitting` expects: rows are
time periods, columns are candidate parameter choices, values are that
choice's return on that day.

## Result 1 — the full 33-year history

Ranking all 16 pairs by total return over the whole sample:

```
MA50_250    3.586
MA50_200    3.452
MA30_200    3.423
MA10_200    3.361
MA50_150    3.329
...
MA30_100    2.402   <- worst
```

`50/250` wins — close to the popular `50/200`, so far so good for the
folklore. But run `probability_of_backtest_overfitting` on the full trial
set:

```python
from quanttro.robustness import probability_of_backtest_overfitting

result = probability_of_backtest_overfitting(trial_df, n_partitions=10)
print(result["pbo"])
```

**Output:**
```
0.5833333333333334
```

**PBO ≈ 0.58.** Across the 252 in-sample/out-of-sample combinatorial
splits the procedure tests, the parameter pair that looked best
in-sample underperformed the median out-of-sample 58% of the time — barely
better than a coin flip. The "winner" is not reliably the winner.

## Result 2 — does the strategy itself have an edge at all?

PBO and the Deflated Sharpe Ratio answer *different* questions, and it's
worth running both. DSR asks: after correcting for the fact that 16
parameter choices were tried, is the winning choice's Sharpe ratio still
distinguishable from zero?

```python
from scipy.stats import skew, kurtosis as kurt
from quanttro.backtest.metrics import sharpe_ratio, deflated_sharpe_ratio

best = trial_df[trial_df.sum().idxmax()]  # MA50_250
sr = sharpe_ratio(best)
dsr = deflated_sharpe_ratio(
    sr, n_trials=16, skewness=skew(best), kurtosis=kurt(best, fisher=False),
    n_observations=len(best),
)
print(sr, dsr)
```

**Output:**
```
0.7731313892170489 1.0
```

DSR saturates at 1.0 — with 8,481 daily observations (33 years), even a
moderate Sharpe ratio becomes statistically indistinguishable from "real"
after multiple-testing correction. **This is the nuance PBO catches and
DSR doesn't:** the moving-average-crossover *approach* has a genuine
long-run statistical edge over this sample, but that says nothing about
whether *50/250 specifically* (versus any of the other 15 pairs) is the
right choice. DSR answers "is there skill here at all"; PBO answers "is
the specific optimized choice trustworthy." A strategy can pass the first
test and fail the second — which is exactly what happened here.

## Result 3 — a different time window crowns a different winner

Restricting to the most recent 10 years (2016–2026, closer to what a
real trader would actually calibrate on):

```python
recent = trial_df.loc[trial_df.index >= "2016-01-01"]
print(recent.sum().sort_values(ascending=False))
print(probability_of_backtest_overfitting(recent, n_partitions=10)["pbo"])
```

**Output:**
```
MA10_100    1.280   <- new winner
MA50_250    1.263
MA10_200    1.242
...
MA50_100    0.817   <- worst

PBO: 0.579
```

The "best" pair over the last decade is **`10/100`**, not `50/250` and not
the folklore `50/200`. PBO is essentially unchanged (~0.58). The specific
winning parameter pair depends on which slice of history you optimize on —
precisely the instability PBO is designed to detect.

## Takeaway

None of this means moving-average crossovers "don't work" — DSR suggests
there's a real signal in the approach over 33 years of SPY data. It means
that picking a *specific* lookback pair by backtesting several and keeping
the best one is, by this measure, close to a coin flip on which pair was
actually best versus which pair got lucky in-sample. Treat a single
"optimal" parameter pair quoted without an overfitting check — on this
strategy or any other — with real skepticism; this is exactly the kind of
check `quanttro.robustness` exists to make routine instead of occasional.

Full code for this case study: the snippets above are copy-pasteable as-is
against `pip install quanttro`.
