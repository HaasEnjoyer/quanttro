# `quanttro.diagnostics` and `quanttro.reports`

This document covers both modules together, because `reports.generate` is built
directly on top of `diagnostics.anomalies`:

- **`quanttro.diagnostics`** — functions that automatically detect BIST-specific
  data/market anomalies (limit-up/limit-down streaks, outlier stocks) and regime
  breaks in walk-forward analyses. It turns patterns found through manual exploratory
  analysis (such as the limit-streak anomaly in KONTR, or the regime break in the
  weekly volume-return relationship) into reusable tools.
- **`quanttro.reports`** — two engines that generate automated summary reports
  (Markdown/HTML) for a symbol or a return series: `generate_symbol_report`
  (price/volume + anomaly summary) and `tearsheet` (a QuantStats-style performance
  scorecard).

All of them are also accessible via `import quanttro as bq` (e.g.
`bq.detect_limit_streaks`, `bq.detect_regime_break`, `bq.tearsheet`,
`bq.generate_symbol_report`); `rolling_correlation_stability` is only available in the
`quanttro.diagnostics` subpackage/directly in `quanttro.diagnostics.regime` (it has
not been exposed at the top-level `bq` namespace).

For methodological pitfalls (what to watch out for, which assumptions can break), see
`docs/KNOWN_PITFALLS.md`.

---

## 1. Anomaly Detection (`quanttro.diagnostics.anomalies`)

**File:** `quanttro/diagnostics/anomalies.py`

This module turns findings discovered manually in the KONTR stock — such as the
"8-day limit-up streak" and a "limit-up+limit-down ratio 3.7x higher than the rest" —
into general, reusable functions. The default threshold `LIMIT_THRESHOLD = 0.095`
corresponds to BIST's 10% limit-up/limit-down (tavan/taban) band (the same logic as
`LIMIT_THRESHOLD` in `feature_engineering.py` — 9.5% rather than 10% is used to leave a
margin for rounding/slippage).

**Important:** these functions make no claim of "predictability." A high
autocorrelation/limit-streak frequency can be a sign of manipulation or distress in
low-float periods (see `docs/KNOWN_PITFALLS.md`). This module only answers the
automated question "does this pattern exist, and how severe is it" — interpretation is
left to the user.

The examples below were run against real `data/daily/KONTR.csv` data.

### `detect_limit_streaks(returns, threshold=LIMIT_THRESHOLD, min_streak=2)`

Finds streaks of consecutive limit-up (+limit) or limit-down (-limit) days. Written to
automatically detect events like the "8-day limit-up streak" found manually in the
KONTR analysis: it returns all blocks in the `returns` series where the same direction
(always limit-up or always limit-down) occurs consecutively, of at least `min_streak`
length.

**Parameters:**
- `returns` (`pd.Series`): A daily (or other periodic) return series, with a `DatetimeIndex`.
- `threshold` (float): Days where `|return| >= threshold` are counted as "at the limit." Default `LIMIT_THRESHOLD = 0.095`.
- `min_streak` (int): The minimum number of consecutive days to report. Default `2`.

**Returns:** `pd.DataFrame` — columns: `start_date, end_date, length, direction`
(`"tavan"` | `"taban"`, i.e. "limit-up" | "limit-down"), sorted in descending order of
length. If no streaks are found, an empty DataFrame (but with the correct columns) is
returned.

**Example:**
```python
import pandas as pd
from quanttro.diagnostics.anomalies import detect_limit_streaks, LIMIT_THRESHOLD

df = pd.read_csv("data/daily/KONTR.csv", index_col=0, parse_dates=True)
returns = df["Close"].pct_change()

streaks = detect_limit_streaks(returns, min_streak=2)
print(streaks.head(5))
print("toplam seri sayisi:", len(streaks))
```

**Output:**
```
  start_date   end_date  length direction
0 2020-10-20 2020-10-30       8     tavan
1 2021-01-26 2021-02-03       7     tavan
2 2026-07-07 2026-07-14       6     tavan
3 2026-06-18 2026-06-25       6     taban
4 2022-11-11 2022-11-16       4     tavan
toplam seri sayisi: 30
```

KONTR's manually discovered "8-day limit-up streak" (2020-10-20 → 2020-10-30) is
correctly auto-detected by the function (and ranked at the top as the longest streak);
30 separate consecutive limit-up/limit-down streaks were found in total.

### `limit_frequency_by_year(returns, threshold=LIMIT_THRESHOLD)`

Computes the ratio of limit-up/limit-down days on a per-year basis. Written to
automatically produce the yearly limit-up/limit-down ratios found in the KONTR
analysis (e.g. "24.5% in 2020, 11% in 2026"). Used to see how a stock's regime has
changed over time (for example, comparing a post-IPO low-float period against a later
high-float period).

**Parameters:**
- `returns` (`pd.Series`): A daily return series, with a `DatetimeIndex`.
- `threshold` (float): The threshold for counting a limit-up/limit-down day. Default `LIMIT_THRESHOLD`.

**Returns:** `pd.DataFrame` — columns: `year, n_gun, n_tavan, n_taban, tavan_orani,
taban_orani, toplam_limit_orani` (i.e. year, number of days, number of limit-up days,
number of limit-down days, limit-up ratio, limit-down ratio, total limit ratio). Sorted
by year in ascending order.

**Example:**
```python
from quanttro.diagnostics.anomalies import limit_frequency_by_year

yearly = limit_frequency_by_year(returns)
print(yearly)
```

**Output:**
```
   year  n_gun  n_tavan  n_taban  tavan_orani  taban_orani  toplam_limit_orani
0  2020     52       13        1     0.250000     0.019231            0.269231
1  2021    249       27        6     0.108434     0.024096            0.132530
2  2022    252       20        5     0.079365     0.019841            0.099206
3  2023    249       16        3     0.064257     0.012048            0.076305
4  2024    250        7        2     0.028000     0.008000            0.036000
5  2025    251       11        3     0.043825     0.011952            0.055777
6  2026    191       21       18     0.109948     0.094241            0.204188
```

In 2020 (the post-IPO low-float period) the limit-up ratio was 25%, falling to 2.8% by
2024 (the regime normalized), but in 2026 the combined limit-up+limit-down ratio jumped
back up to 20.4% — a jump of this kind is a candidate regime change that can also be
confirmed with `quanttro.diagnostics.regime.detect_regime_break`.

### `flag_outlier_stocks(symbol_stats, combined_rate_threshold=0.05, return_col="ret_1", threshold=LIMIT_THRESHOLD, lookback_bars=252, relative_multiplier=3.0)`

Flags stocks whose recent limit-up+limit-down ratio is abnormally high. Generalizes the
logic that sets KONTR apart from the rest by ~3.7x: it computes each stock's
limit-up+limit-down ratio over the last `lookback_bars` (default ~1 year) days, and
flags a stock as an "outlier" if either (a) this ratio exceeds the absolute threshold
(`combined_rate_threshold`), or (b) it is more than `relative_multiplier` times the
median across all stocks.

**Parameters:**
- `symbol_stats` (`dict[str, pd.DataFrame]`): `{"KONTR": df, "THYAO": df, ...}` — each
  df must contain at least the `return_col` column (daily return), sorted by a
  `DatetimeIndex`.
- `combined_rate_threshold` (float): The absolute threshold — a stock whose
  limit-up+limit-down ratio exceeds this is flagged directly. Default `0.05`.
- `return_col` (str): The name of the return column (`"ret_1"` in feature tables). Default `"ret_1"`.
- `threshold` (float): The threshold for counting a limit-up/limit-down day. Default `LIMIT_THRESHOLD`.
- `lookback_bars` (int): The number of recent bars to evaluate. Default `252` (~1 trading year).
- `relative_multiplier` (float): Determines how many times the median a stock must exceed to also be considered an "outlier." Default `3.0`.

**Returns:** `list[str]` — an alphabetically sorted list of flagged symbols. Symbols
that cannot be computed (no data/column) are silently skipped.

**Example:**
```python
import numpy as np
from quanttro.diagnostics.anomalies import flag_outlier_stocks

rng = np.random.default_rng(1)
normal_returns = pd.Series(rng.normal(0, 0.02, size=len(returns)), index=returns.index)
symbol_stats = {
    "KONTR": pd.DataFrame({"ret_1": returns}),
    "NORMAL1": pd.DataFrame({"ret_1": normal_returns}),
    "NORMAL2": pd.DataFrame({"ret_1": normal_returns * 0.9}),
}
flagged = flag_outlier_stocks(symbol_stats, lookback_bars=252)
print("flagged:", flagged)
```

**Output:**
```
flagged: ['KONTR']
```

When comparing the real KONTR returns against two synthetic "normal" (low-volatility,
non-limit) return series, only KONTR's limit-up+limit-down ratio over the last 252 bars
exceeds the thresholds and gets flagged — the function genuinely can distinguish what
is "abnormal."

---

## 2. Regime Detection (`quanttro.diagnostics.regime`)

**File:** `quanttro/diagnostics/regime.py`

A concrete example found in the walk-forward analysis in `analyze_weekly_volume.py`:
the `genel_fark` ("overall difference") column is consistently negative across the
first 4 folds, then flips to positive in the last fold (2022-06-10 - 2026-09-25). This
module was written to generally evaluate this kind of "did the last fold flip sign
relative to the previous folds" question, and how stable the correlation between two
series is over time.

**Key lesson** (see `docs/KNOWN_PITFALLS.md`): collapsing walk-forward folds into a
single "average" number can hide the underlying regime change. These functions are
meant to make that change visible.

### `detect_regime_break(fold_results, value_col, sign_flip_threshold=1)`

Detects whether the last fold(s) of walk-forward results flipped sign.

**Parameters:**
- `fold_results` (`pd.DataFrame`): Each row is one fold; must be sorted in
  chronological order (the first row is the oldest fold). For example, the output of
  `analyze_weekly_volume.walk_forward_regime_check()` or `validate_volume_dow_rule.run()`.
- `value_col` (str): The metric column to examine (e.g. `"genel_fark"`, `"fark"`, `"momentum_fark"`).
- `sign_flip_threshold` (int): How many of the last folds are split off as the "recent"
  group and compared against the mean of the preceding folds. Default `1` (only the
  very last fold).

**Returns:** `dict` —
```
{
    "n_folds": int,            # total number of folds
    "prior_mean": float,       # mean of value_col over the prior folds
    "recent_mean": float,      # mean of value_col over the recent fold(s)
    "sign_flipped": bool,      # whether the sign flipped
    "t_stat": float,           # t-statistic between prior and recent (Welch's t-test)
    "p_value": float,          # the p-value for this difference
    "warning": str | None,     # an explanatory message if the sign flipped, otherwise None
}
```

If there are not enough folds (`n_folds < sign_flip_threshold + 2`), all numeric fields
are returned as `NaN`/`False` and the `warning` field explains the insufficiency (no
exception is raised). If the `recent` group consists of a single value
(`sign_flip_threshold=1` with only 1 "recent" fold), a formal p-value cannot be
computed, so `t_stat` is instead computed as a single-sample deviation measure relative
to the prior distribution, and `p_value` stays `NaN`.

**Example:**
```python
import pandas as pd
from quanttro.diagnostics.regime import detect_regime_break

# A synthetic 5-fold walk-forward result: "genel_fark" negative in the first 4 folds, positive in the last
fold_results = pd.DataFrame({
    "fold": [1, 2, 3, 4, 5],
    "genel_fark": [-0.012, -0.018, -0.009, -0.015, 0.021],
})
result = detect_regime_break(fold_results, value_col="genel_fark", sign_flip_threshold=1)
for k, v in result.items():
    print(f"{k}: {v}")
```

**Output:**
```
n_folds: 5
prior_mean: -0.0135
recent_mean: 0.021
sign_flipped: True
t_stat: 17.81572339255412
p_value: nan
warning: UYARI: son 1 fold'da 'genel_fark' isareti onceki fold'lara gore TERS DONDU (-0.01350 -> 0.02100). Tum ornegin tek bir 'ortalama' sonucuna guvenmeyin - rejim kirilmasi olasi (bkz. docs/KNOWN_PITFALLS.md: 'haftalik hacim-getiri iliskisi rejim-bagimlidir').
```

Since `sign_flip_threshold=1`, the "recent" group is a single value (the last fold,
`0.021`), which is why `p_value` stays `nan` (the special case explained above);
`t_stat` nonetheless shows the magnitude of the deviation relative to the prior
distribution (17.8 standard errors), and the sign flip is correctly captured. (Note:
the `warning` string itself is returned as generated by the library, in Turkish — it
translates as: "WARNING: in the last 1 fold(s), the sign of 'genel_fark' REVERSED
relative to the prior folds (-0.01350 -> 0.02100). Do not rely on a single 'average'
result for the whole sample — a regime break is likely (see docs/KNOWN_PITFALLS.md:
'the weekly volume-return relationship is regime-dependent').")

### `rolling_correlation_stability(x, y, window=252)`

Computes the rolling correlation between two series over time. A single "full-period"
correlation coefficient can hide the fact that the underlying relationship has been
strengthening, weakening, or even reversing direction over time. The std()/trend of the
result itself can also be used to produce a numeric answer to "how stable" the
relationship is (a declining or sign-flipping rolling correlation is an early warning
sign of a regime break).

**Parameters:**
- `x`, `y` (`pd.Series`): The two series to compare (can have the same or different
  indexes — they are first aligned on their intersecting index).
- `window` (int): The length of the rolling window. Default `252` (~1 trading year).

**Returns:** `pd.Series` — the Pearson correlation over the last `window` bars at every
point, over the intersecting index of `x` and `y`. Returns `NaN` before the window is
filled.

**Example:**
```python
import numpy as np
import pandas as pd
from quanttro.diagnostics.regime import rolling_correlation_stability

rng = np.random.default_rng(3)
idx = pd.date_range("2021-01-01", periods=600, freq="B")
x = pd.Series(rng.normal(0, 1, len(idx)), index=idx).cumsum()
# y is positively correlated with x in the first half, independent in the second half
y1 = x.iloc[:300] * 0.8 + rng.normal(0, 1, 300)
y2 = pd.Series(rng.normal(0, 5, len(idx) - 300), index=idx[300:])
y = pd.concat([y1, y2])

corr_series = rolling_correlation_stability(x, y, window=100)
print("Ilk donem (x-y yuksek korele) ortalama rolling corr (200-300 bar):", corr_series.iloc[200:300].mean())
print("Son donem (bagimsiz) ortalama rolling corr (500-600 bar):", corr_series.iloc[500:600].mean())
```

**Output:**
```
Ilk donem (x-y yuksek korele) ortalama rolling corr (200-300 bar): 0.9611442136465529
Son donem (bagimsiz) ortalama rolling corr (500-600 bar): -0.007155986319047427
```

The function correctly distinguishes the high correlation in the first half of the
series (~0.96, the period where y is constructed to depend on x) from the near-zero
correlation in the second half (~-0.01, the period where y reverts to an independent
random walk) — exactly a demonstration of how a regime break shows up in a rolling
correlation.

---

## 3. Report Generation (`quanttro.reports`)

### 3.1 `generate_symbol_report`

**File:** `quanttro/reports/generate.py`

The goal is to produce a short, readable report file containing a price/volume summary
plus anomaly detection (`quanttro.diagnostics.anomalies`) for each stock, without
repeating the manually written analysis scripts (`analyze_autocorrelation.py`,
`analyze_volume_calendar.py`, etc.). It is not a complex reporting engine — it's a
simple string-template approach designed for quickly reviewing a single stock.

#### `generate_symbol_report(symbol, df, output_path, format="markdown")`

Produces a report file for `symbol` containing a price/volume summary and anomaly detection.

**Parameters:**
- `symbol` (str): The symbol name shown in the report title (e.g. `"KONTR"`, `"THYAO"`).
- `df` (`pd.DataFrame`): A price series with a `DatetimeIndex`. Supports both raw OHLCV
  format (`"Close"`, `"Volume"` — the output of `fetch_data.py`) and feature-table
  format (`"close"`, `"ret_1"`, ... — the output of `feature_engineering.py`).
- `output_path` (str): The path the report file is written to. Parent directories are created automatically if they don't exist.
- `format` (str): `"markdown"` (default, `.md` content) or `"html"` (`.html` content). Raises `ValueError` for any other value.

**Returns:** `None` — the result is written directly to `output_path`.

**Example:**
```python
import pandas as pd
from quanttro.reports.generate import generate_symbol_report

df = pd.read_csv("data/daily/KONTR.csv", index_col=0, parse_dates=True)
generate_symbol_report("KONTR", df, "/tmp/KONTR_report.md", format="markdown")
print(open("/tmp/KONTR_report.md").read())
```

**Output** (the full content of the generated file, 1713 characters; opening sections):
```
# KONTR - Ozet ve Anomali Raporu

*Bu rapor `quanttro.reports.generate.generate_symbol_report` tarafindan otomatik uretildi.*

## Fiyat Ozeti

- Donem: 2020-10-19 -> 2026-10-02 (1495 bar)
- Son kapanis: 2.34
- Donem getirisi: 461.87%
- Ortalama gunluk getiri: 0.2223%
- Gunluk getiri std: 4.5622%
- Yaklasik yillik volatilite (std * sqrt(252)): 72.42%

## Hacim Ozeti

- Ortalama hacim: 31,747,376
- Son 20 bar ortalama hacim: 5,421,501

## Tavan/Taban Serisi Tespiti

(`detect_limit_streaks`, esik = %9.5, min. 2 ardisik gun)

Toplam **30** seri bulundu (en uzunu: **8 gun**).

| Baslangic | Bitis | Uzunluk (gun) | Yon |
|---|---|---|---|
| 2020-10-20 | 2020-10-30 | 8 | tavan |
...

## Yillik Tavan/Taban Orani

(`limit_frequency_by_year`)

| Yil | Gun Sayisi | Tavan Orani | Taban Orani | Toplam |
|---|---|---|---|---|
| 2020 | 52 | 25.00% | 1.92% | 26.92% |
...
```

*(Note: the report content itself is generated by the library in Turkish — e.g. "KONTR
- Summary and Anomaly Report," "Price Summary," "Period," "Last close," "Period
return," "Average daily return," "Daily return std," "Approximate annualized
volatility," "Volume Summary," "Average volume," "Average volume over the last 20
bars," "Limit-Up/Limit-Down Streak Detection," "threshold," "min. N consecutive days,"
"N streaks found in total (longest: N days)," "Start / End / Length (days) / Direction"
table headers, "Yearly Limit-Up/Limit-Down Ratio," and "Year / Number of Days / Limit-Up
Ratio / Limit-Down Ratio / Total" table headers — these strings are produced by the
report-generation code itself and are shown here exactly as emitted.)*

The report automatically merges both the `detect_limit_streaks` and
`limit_frequency_by_year` results into a single readable file — no manual analysis required.

### 3.2 `tearsheet` (`quanttro.reports.tearsheet`)

**File:** `quanttro/reports/tearsheet.py`

A comprehensive, QuantStats-style performance scorecard generator. This module
directly imports and uses the metric functions that are definitely present in
`quanttro.backtest.metrics` (`sharpe_ratio`, `sortino_ratio`, `max_drawdown`,
`calmar_ratio`, `hit_rate`, `profit_factor`). The other (optional) metrics
(`omega_ratio`, `tail_ratio`, `value_at_risk`, `conditional_value_at_risk`, `alpha_beta`,
`capture_ratio`, `kelly_criterion`, `risk_of_ruin`, `ulcer_index`, `common_sense_ratio`,
`avg_win`, `avg_loss`, `payoff_ratio`, `recovery_factor`, `gain_to_pain_ratio`) are
called safely via `getattr(metrics_module, "...", None)` — if the function isn't found,
or the call fails for any reason, that row is simply dropped from the report, and the
report never raises an exception.

#### `tearsheet(returns, benchmark_returns=None, output_path=None, format="html")`

Generates a comprehensive, QuantStats-style performance scorecard (tearsheet): a
summary table (total return, CAGR, annualized volatility, Sharpe, Sortino, Calmar,
maximum drawdown, hit rate, profit factor), additional risk/performance metrics when
available, a benchmark comparison when provided (alpha/beta, capture ratio), annual
returns, a monthly-return heatmap, and the largest drawdown periods.

**Parameters:**
- `returns` (`pd.Series`): A simple (arithmetic) return series with a `DatetimeIndex`.
- `benchmark_returns` (`pd.Series | None`): If provided, a "Benchmark Comparison"
  section is added over the common date range (correlation, benchmark total return,
  and alpha/beta, capture ratio if available).
- `output_path` (`str | None`): If provided, the report is written to this path (parent
  directories created automatically). If not provided, only the string is returned.
- `format` (str): `"html"` (default) or `"markdown"`. Raises `ValueError` for any other value.

**Returns:** `str` — the generated report content (returned whether or not `output_path` is given).

**Example** (using real KONTR returns):
```python
import pandas as pd
from quanttro.reports.tearsheet import tearsheet

df = pd.read_csv("data/daily/KONTR.csv", index_col=0, parse_dates=True)
returns = df["Close"].pct_change().dropna()

content = tearsheet(returns, format="markdown")
print(content)
```

**Output** (opening sections, 2435 characters in total):
```
# Performans Karnesi (Tearsheet)

*Bu rapor `quanttro.reports.tearsheet.tearsheet` tarafindan otomatik uretildi.*

## Ozet

| Metrik | Deger |
|---|---|
| Toplam Getiri | 461.87% |
| Yillik Getiri (CAGR) | 33.80% |
| Yillik Volatilite | 72.42% |
| Sharpe Orani | 0.773 |
| Sortino Orani | 1.193 |
| Calmar Orani | 0.345 |
| Maksimum Drawdown | -97.95% |
| Hit Rate | 46.18% |
| Profit Factor | 1.147 |

## Ek Risk/Performans Metrikleri

| Metrik | Deger |
|---|---|
| Omega Ratio | 1.1475 |
| Tail Ratio | 1.6199 |
| VaR (gunluk, %95) | -6.14% |
| CVaR (gunluk, %95) | -9.21% |
| Kelly Kriteri | 0.0443 |
| Risk of Ruin | 100.00% |
| Ulcer Index | 102.30% |
| Common Sense Ratio | 1.8588 |
| Ortalama Kazanc (gunluk) | 3.74% |
| Ortalama Kayip (gunluk) | -2.90% |
| Payoff Ratio | 1.2888 |
| Recovery Factor | -0.1112 |
| Gain-to-Pain Ratio | 0.1475 |
...
```

*(Note: as with the symbol report above, the generated report's own text is produced
by the library in Turkish — "Performance Scorecard (Tearsheet)," "Summary," "Metric /
Value" headers, "Total Return," "Annual Return (CAGR)," "Annual Volatility," "Sharpe
Ratio," "Sortino Ratio," "Calmar Ratio," "Maximum Drawdown," "Hit Rate," "Profit
Factor," "Additional Risk/Performance Metrics," "VaR (daily, 95%)," "CVaR (daily,
95%)," "Kelly Criterion," "Risk of Ruin," "Ulcer Index," "Common Sense Ratio," "Average
Gain (daily)," "Average Loss (daily)," "Payoff Ratio," "Recovery Factor,"
"Gain-to-Pain Ratio" — shown here verbatim as emitted by the report generator.)*

In the currently installed version of this library, all of the optional metrics in
`quanttro.backtest.metrics` are already present, so the "Additional Risk/Performance
Metrics" section appears in full; at a development stage where these functions haven't
been added yet, the same report continues to be generated without error, with those
rows silently dropped.

#### `monthly_returns_table(returns)`

Converts a return series into a year × month pivot table. Each cell is the compounded
total return of the simple (daily) returns within that year-month: `(1 + r).prod() - 1`.

**Parameters:**
- `returns` (`pd.Series`): A simple return series with a `DatetimeIndex`.

**Returns:** `pd.DataFrame` — index = year (int), columns = 1..12 (month number); cells
with no data for that month-year are `NaN`.

**Example:**
```python
from quanttro.reports.tearsheet import monthly_returns_table

mt = monthly_returns_table(returns)
print(mt.head(3))
```

**Output:**
```
month        1         2         3   ...        10        11        12
year                                 ...                              
2020        NaN       NaN       NaN  ...  1.133070  0.025324  0.418072
2021   1.489381  0.268772 -0.027572  ...  0.209744 -0.032082 -0.083216
2022   0.789231  0.036113  0.136930  ...  0.683533  0.667855 -0.032987
```

#### `rolling_sharpe(returns, window=252, periods_per_year=252)`

A rolling-window Sharpe ratio series. For performance, instead of repeatedly calling
`metrics.sharpe_ratio` on every window, the formula is computed directly from the
rolling mean/std:

```
rolling_sharpe_t = mean(r[t-window+1:t+1]) / std(r[t-window+1:t+1], ddof=1) * sqrt(periods_per_year)
```

The risk-free rate is not included (treated as 0) — the goal is to observe the Sharpe
fluctuation over time; `metrics.sharpe_ratio` should be used for a point estimate.

**Parameters:**
- `returns` (`pd.Series`): A simple return series.
- `window` (int): The rolling window size (number of periods). Default `252`.
- `periods_per_year` (int): Periods per year for annualization. Default `252`.

**Returns:** `pd.Series` — a Sharpe series with the same index as `returns`, where the
first `window - 1` values (and any points where the within-window std is ~0) are `NaN`
(`name="rolling_sharpe"`).

**Example:**
```python
from quanttro.reports.tearsheet import rolling_sharpe

rs = rolling_sharpe(returns, window=60)
print(rs.dropna().tail(5))
```

**Output:**
```
Date
2026-09-28   -1.394407
2026-09-29   -1.986716
2026-09-30   -2.543632
2026-10-01   -2.531607
2026-10-02   -2.715130
Name: rolling_sharpe, dtype: float64
```

KONTR's sharply negative 60-day rolling Sharpe in its most recent period (consistent
with the jump in the limit-down ratio seen in the `limit_frequency_by_year` result for
2026 above) directly reflects the unrecovered decline in the price series.

#### `drawdown_periods(returns, top_n=5)`

Returns the largest `top_n` drawdown periods in the return series as a table. Each row
defines one drawdown period:

- `start`: the peak date at which the period begins — the previous bar at which the
  running max was last updated, right before the decline begins.
- `valley`: the date of the lowest (trough) point within that period.
- `recovery`: the first date at which the curve again exceeds the previous peak (its
  value on the `start` date); `None` if the period had not recovered by the end of the report.
- `depth`: the worst (negative) percentage decline of that period.

Periods are sorted by `depth` (worst to least severe).

**Parameters:**
- `returns` (`pd.Series`): A simple return series with a `DatetimeIndex`.
- `top_n` (int): The number of deepest drawdown periods to return. Default `5`.

**Returns:** `pd.DataFrame` — columns: `start`, `valley`, `recovery`, `depth`.

**Example:**
```python
from quanttro.reports.tearsheet import drawdown_periods

dd = drawdown_periods(returns, top_n=3)
print(dd)
```

**Output:**
```
       start     valley   recovery     depth
0 2023-09-25 2026-09-30        NaT -0.979464
1 2021-03-08 2021-04-21 2022-01-27 -0.516311
2 2022-11-16 2023-05-15 2023-07-24 -0.508349
```

Since the deepest drawdown period (`-97.95%`, starting 2023-09-25 and lasting through
the end of the data) has not yet recovered, `recovery` shows `NaT` (pandas's missing
date value) — the function's docstring conceptually defines this case as `None`, but
in practice, because the `pd.DataFrame` column is coerced to `datetime64` (since the
other rows contain real `Timestamp` values), `None` is automatically converted to `NaT`,
and the Markdown/HTML rendering functions print `"NaT"` in this case too (rather than
`"(not yet recovered)"`) — a minor inconsistency that can also be observed in the
actual report files produced via `output_path`.
