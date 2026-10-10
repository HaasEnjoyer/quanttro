# `quanttro.features`

The `quanttro.features` module generates features from OHLCV data for use
in modeling/backtesting. It is split into four submodules:

- **`core`**: basic price/volume features (lagged returns, volume
  z-score, volatility, ATR, limit-up/limit-down flag, forward return target).
- **`technical`**: classic technical analysis indicators (RSI, MACD, Bollinger,
  Stochastic, OBV, Momentum).
- **`calendar`**: calendar/time-based features (day of week, turn-of-month,
  Monday/Tuesday rule).
- **`cross_sectional`**: multi-stock (panel) based relative strength/ranking
  features - for stripping out market-wide/sector-wide common trends.

The `core`, `technical`, `calendar` functions operate on a **single stock**
(input: `pd.Series`, usually one of the `Close`/`High`/`Low`/`Volume`
columns) and preserve the SAME index as the input; `NaN` values (e.g.
before a rolling window has warmed up) are left in deliberately - deciding
which rows to drop (`dropna`) is the caller's responsibility (a deliberate
choice to prevent look-ahead leakage). The `cross_sectional` functions, on
the other hand, expect **multi-stock** data in "wide" format (index = time,
columns = symbol).

High-level shortcut for quickly generating single-stock features:
`quanttro.make_features(df, include=("core", "technical", "calendar"))`
(see the end of this file).

---

## `core` - basic price/volume features (`quanttro.features.core`)

This module is the generalized form of the original logic in
`feature_engineering.py` at the project root. Two important design
decisions were carried over from there unchanged:

- **Winsorize (RETURN_CLIP):** returns are clipped to the `[-clip, +clip]`
  range to prevent market-wide shock bars from distorting the return
  distribution.
- **Limit-up/limit-down threshold (LIMIT_THRESHOLD = 0.095):** on BIST,
  daily price movement is practically capped at +-10% due to limit-up
  (tavan) / limit-down (taban) rules; this threshold flags bars close to
  that boundary as "limit-up/limit-down" with a safety margin.

Constants: `DEFAULT_LAGS = (1, 5, 10, 20)`, `DEFAULT_RETURN_CLIP = 0.12`,
`DEFAULT_LIMIT_THRESHOLD = 0.095`.

### `lagged_returns(close, lags=(1, 5, 10, 20), clip=0.12)`

Generates percentage return columns for the given lags (number of bars).

**Parameters:**
- `close` (`pd.Series`): closing price series
- `lags` (`Sequence[int]`): the lags to compute, default `(1, 5, 10, 20)`;
  a `ret_{n}` column is generated for each (`close.pct_change(n)`)
- `clip` (`float | None`): winsorize threshold, default `0.12`;
  if `None`, no clipping is applied (raw return)

**Returns:** `pd.DataFrame` - columns in `ret_{n}` format, with the same
index as `close`.

**Example:**
```python
from quanttro.data import fetch
from quanttro.features import core

df = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
close = df["Close"]
lr = core.lagged_returns(close)
print(lr.tail(3))
```

**Output:**
```
               ret_1     ret_5    ret_10    ret_20
Date                                              
2026-10-06 -0.005988 -0.003431 -0.025168 -0.047541
2026-10-07 -0.014630  0.010591 -0.041039 -0.049003
2026-10-08 -0.006987 -0.007853 -0.014731 -0.050125
```

### `volume_zscore(volume, window=20)`

The z-score of volume relative to its own rolling mean/standard deviation
over the last `window` bars: `(volume - rolling_mean) / rolling_std`. Used
to detect abnormally high/low volume bars (for example as an input to the
Monday/Tuesday + high-volume rule together with
`calendar.is_monday_or_tuesday`).

**Parameters:**
- `volume` (`pd.Series`): volume series
- `window` (int): rolling window length, default `20`

**Returns:** `pd.Series`, named `"volume_zscore"`.

**Example:**
```python
vz = core.volume_zscore(df["Volume"])
print(vz.tail(3))
```

**Output:**
```
Date
2026-10-06   -0.777234
2026-10-07   -0.711079
2026-10-08   -1.047592
Name: volume_zscore, dtype: float64
```

### `realized_volatility(returns, window=10)`

Realized volatility of returns over the last `window` bars (sample
standard deviation, `ddof=1`, pandas default).

**Parameters:**
- `returns` (`pd.Series`): return series (e.g. `close.pct_change()`)
- `window` (int): rolling window length, default `10`

**Returns:** `pd.Series`, named `"realized_vol_{window}"`.

**Example:**
```python
ret1 = close.pct_change()
rv = core.realized_volatility(ret1)
print(rv.tail(3))
```

**Output:**
```
Date
2026-10-06    0.017883
2026-10-07    0.018211
2026-10-08    0.015109
Name: realized_vol_10, dtype: float64
```

### `average_true_range(high, low, close, window=14, normalize=True)`

Average True Range (ATR), Wilder's volatility measure.

```
True Range (TR) = max(
    high - low,
    |high - previous_close|,
    |low - previous_close|,
)
ATR = simple moving average of TR over the last `window` bars
```

Consistent with the existing implementation in `feature_engineering.py`,
a simple rolling mean is used rather than Wilder's exponential (smoothed)
average.

**Parameters:**
- `high`, `low`, `close` (`pd.Series`): the respective price series
- `window` (int): rolling window length, default `14`
- `normalize` (bool): if `True` (default), ATR is divided by the closing
  price to make it percentage-based (independent of price level), making
  stocks at different price levels comparable

**Returns:** `pd.Series`, named `"atr_{window}"`.

**Example:**
```python
atr = core.average_true_range(df["High"], df["Low"], close)
print(atr.tail(3))
```

**Output:**
```
Date
2026-10-06    0.029321
2026-10-07    0.027199
2026-10-08    0.026825
Name: atr_14, dtype: float64
```

### `at_price_limit(returns, threshold=0.095)`

BIST limit-up/limit-down flag: flags bars where `|return| >= threshold`
with `1`.

**Parameters:**
- `returns` (`pd.Series`): return series
- `threshold` (float): threshold, default `DEFAULT_LIMIT_THRESHOLD` (`0.095`)

**Returns:** `pd.Series` (`int`, 0/1), named `"at_limit"`.

**Example:**
```python
limit = core.at_price_limit(ret1)
print(limit.tail(3))
print("total limit-up/limit-down bar count:", limit.sum())
```

**Output:**
```
Date
2026-10-06    0
2026-10-07    0
2026-10-08    0
Name: at_limit, dtype: int64
total limit-up/limit-down bar count: 65
```

### `forward_return(close, horizon=1, clip=None)`

The return looking `horizon` bars ahead (target variable / label):
`fwd_ret_t = close_(t+horizon) / close_t - 1`.

**WARNING:** This function deliberately looks into the FUTURE
(`shift(-horizon)`) - it should only be used to produce a training target,
and must NEVER be fed into a model as a "feature" (look-ahead bias /
future leakage).

**Parameters:**
- `close` (`pd.Series`): closing price series
- `horizon` (int): number of bars to look ahead, default `1`
- `clip` (`float | None`): winsorize threshold, default `None` (no clipping)

**Returns:** `pd.Series`, named `"fwd_ret_{horizon}"`.

**Example:**
```python
fwd = core.forward_return(close)
print(fwd.tail(3))
```

**Output:**
```
Date
2026-10-06   -0.014630
2026-10-07   -0.006987
2026-10-08         NaN
Name: fwd_ret_1, dtype: float64
```
(The last row is `NaN` - because the next bar hasn't occurred yet.)

---

## `technical` - technical analysis indicators (`quanttro.features.technical`)

These indicators did NOT exist in `feature_engineering.py`; this module
enriches the existing feature set with standard technical analysis
indicators.

### `rsi(close, window=14)`

Relative Strength Index (RSI), using Wilder's exponential smoothing
method.

```
delta = close.diff()
gain = max(delta, 0), loss = max(-delta, 0)
avg_gain / avg_loss: exponential moving average (ewm) with alpha = 1/window
RS = avg_gain / avg_loss
RSI = 100 - 100 / (1 + RS)
```

If no loss has ever been observed (`avg_loss == 0`), `RSI = 100` is assumed.

**Parameters:**
- `close` (`pd.Series`): closing price series
- `window` (int): default `14`

**Returns:** `pd.Series`, in the `[0, 100]` range, named `"rsi_{window}"`.

**Example:**
```python
from quanttro.features import technical

r14 = technical.rsi(close)
print(r14.tail(3))
```

**Output:**
```
Date
2026-10-06    47.345663
2026-10-07    44.165150
2026-10-08    42.711087
Name: rsi_14, dtype: float64
```

### `macd(close, fast=12, slow=26, signal=9)`

MACD (Moving Average Convergence Divergence).

```
macd      = EMA(close, fast) - EMA(close, slow)
signal    = EMA(macd, signal)
histogram = macd - signal
```

EMAs are computed with `adjust=False` (the standard "online" exponential
moving average definition).

**Parameters:**
- `close` (`pd.Series`)
- `fast` (int): default `12`
- `slow` (int): default `26`
- `signal` (int): default `9`

**Returns:** `pd.DataFrame` - columns `"macd"`, `"signal"`, `"histogram"`.

**Example:**
```python
m = technical.macd(close)
print(m.tail(3))
```

**Output:**
```
                macd    signal  histogram
Date                                     
2026-10-06 -2.903779 -3.617050   0.713271
2026-10-07 -3.026542 -3.498949   0.472406
2026-10-08 -3.247778 -3.448715   0.200936
```

### `bollinger_bands(close, window=20, n_std=2.0)`

Bollinger Bands: bands at +-`n_std` standard deviations around a simple
moving average (SMA).

```
middle = rolling_mean(close, window)
upper  = middle + n_std * rolling_std(close, window)
lower  = middle - n_std * rolling_std(close, window)
%b     = (close - lower) / (upper - lower)
```

`%b`: the price's position within the band (0 = pinned to the lower band,
1 = pinned to the upper band, 0.5 = exactly at the mean).

**Parameters:**
- `close` (`pd.Series`)
- `window` (int): default `20`
- `n_std` (float): default `2.0`

**Returns:** `pd.DataFrame` - columns `"upper"`, `"middle"`, `"lower"`, `"%b"`.

**Example:**
```python
bb = technical.bollinger_bands(close)
print(bb.tail(3))
```

**Output:**
```
                 upper    middle       lower        %b
Date                                                  
2026-10-06  304.684150  291.0625  277.440850  0.479353
2026-10-07  303.261179  290.3250  277.388821  0.342496
2026-10-08  302.064048  289.5750  277.085952  0.286813
```

### `stochastic_oscillator(high, low, close, window=14, d_window=3)`

Stochastic Oscillator: the percentage position of the close within the
high/low range over the last `window` bars.

```
%K = 100 * (close - lowest_low) / (highest_high - lowest_low)
%D = simple moving average of %K over `d_window` bars (signal line)
```

**Parameters:**
- `high`, `low`, `close` (`pd.Series`)
- `window` (int): default `14`
- `d_window` (int): default `3`

**Returns:** `pd.DataFrame` - columns `"%K"`, `"%D"`.

**Example:**
```python
sto = technical.stochastic_oscillator(df["High"], df["Low"], close)
print(sto.tail(3))
```

**Output:**
```
                   %K         %D
Date                            
2026-10-06  58.119658  70.122283
2026-10-07  18.518519  51.014224
2026-10-08  10.843373  29.160517
```

### `on_balance_volume(close, volume)`

On-Balance Volume (OBV): the cumulative sum of direction-signed volume.

```
OBV_t = OBV_(t-1) + volume_t   if close_t > close_(t-1)
OBV_t = OBV_(t-1) - volume_t   if close_t < close_(t-1)
OBV_t = OBV_(t-1)              if close_t == close_(t-1)
```

For the first bar, since there is no previous close, the direction is
assumed to be `0` (OBV starts at `0`).

**Parameters:**
- `close`, `volume` (`pd.Series`)

**Returns:** `pd.Series`, named `"obv"`.

**Example:**
```python
obv = technical.on_balance_volume(close, df["Volume"])
print(obv.tail(3))
```

**Output:**
```
Date
2026-10-06    5.626080e+11
2026-10-07    5.625810e+11
2026-10-08    5.625605e+11
Name: obv, dtype: float64
```

### `momentum(close, window=10)`

Price momentum: the DIFFERENCE (not percentage) relative to the close
`window` bars ago: `momentum_t = close_t - close_(t-window)`.

**Parameters:**
- `close` (`pd.Series`)
- `window` (int): default `10`

**Returns:** `pd.Series`, named `"momentum_{window}"`.

**Example:**
```python
mom = technical.momentum(close)
print(mom.tail(3))
```

**Output:**
```
Date
2026-10-06    -7.50
2026-10-07   -12.25
2026-10-08    -4.25
Name: momentum_10, dtype: float64
```

---

## `calendar` - calendar/time features (`quanttro.features.calendar`)

Generalizes the `day_of_week` / `is_turn_of_month` / `hour_of_day` logic
from `feature_engineering.py`. Also adds the calendar leg
(`is_monday_or_tuesday`) of the "Monday/Tuesday + high volume -> different
forward return" finding, which was found to be significant for BIST in the
research done in this project (`validate_volume_dow_rule.py`).

### `day_of_week_features(index)`

Generates both the raw category (0=Monday .. 6=Sunday) and one-hot encoded
columns (`"dow_0"`..`"dow_6"`) for day of week. All 7 day columns are
ALWAYS generated (even if that day never appears in the dataset), so that
the column schema stays fixed across different date ranges.

**Parameters:**
- `index` (`pd.DatetimeIndex`)

**Returns:** `pd.DataFrame` - `"day_of_week"` (0-6) + `"dow_0"`..`"dow_6"` (int, one-hot).

**Example:**
```python
from quanttro.features import calendar as feat_calendar

dow = feat_calendar.day_of_week_features(df.index)
print(dow.tail(3))
```

**Output:**
```
            day_of_week  dow_0  dow_1  dow_2  dow_3  dow_4  dow_5  dow_6
Date                                                                    
2026-10-06            1      0      1      0      0      0      0      0
2026-10-07            2      0      0      1      0      0      0      0
2026-10-08            3      0      0      0      1      0      0      0
```

### `turn_of_month_flag(index, n_days=3)`

Turn-of-month flag: `1` if within the first `n_days` days OR the last
`n_days` days of the month, otherwise `0`. With the default `n_days=3`,
this produces the SAME result as the original expression in
`feature_engineering.py`:
`(day_of_month <= 3) | (day_of_month >= days_in_month - 2)`.

**Parameters:**
- `index` (`pd.DatetimeIndex`)
- `n_days` (int): default `3`

**Returns:** `pd.Series` (`int`, 0/1), named `"is_turn_of_month"`.

**Example:**
```python
tom = feat_calendar.turn_of_month_flag(df.index)
print(tom.tail(5))
```

**Output:**
```
Date
2026-10-02    1
2026-10-05    0
2026-10-06    0
2026-10-07    0
2026-10-08    0
Name: is_turn_of_month, dtype: int64
```
(2026-10-02 is `1` because it falls within the first 3 days of the month.)

### `hour_of_day_features(index)`

Hour-of-day information (0-23) for intraday data. Meaningless for daily
data (returns a single constant value, usually `0`).

**Parameters:**
- `index` (`pd.DatetimeIndex`)

**Returns:** `pd.Series` (int), named `"hour_of_day"`.

**Example:**
```python
hod = feat_calendar.hour_of_day_features(df.index)
print(hod.tail(3))
```

**Output:**
```
Date
2026-10-06    0
2026-10-07    0
2026-10-08    0
Name: hour_of_day, dtype: int32
```
(Since this is daily data, all values are `0` - meaningful values are
obtained on hourly/5-minute data.)

### `is_monday_or_tuesday(index)`

`1` if Monday (0) or Tuesday (1), otherwise `0`. This is the calendar leg
of the "Monday/Tuesday + high volume -> different forward return" rule
found in this project's walk-forward validation
(`validate_volume_dow_rule.py`, `TARGET_DOWS = {0: Monday, 1: Tuesday}`);
it is expected to be used TOGETHER with `core.volume_zscore` (the high
volume condition) to produce a meaningful signal - on its own, this
function only carries the day information.

**Parameters:**
- `index` (`pd.DatetimeIndex`)

**Returns:** `pd.Series` (`int`, 0/1), named `"is_monday_or_tuesday"`.

**Example:**
```python
mt = feat_calendar.is_monday_or_tuesday(df.index)
print(mt.tail(5))
```

**Output:**
```
Date
2026-10-02    0
2026-10-05    1
2026-10-06    1
2026-10-07    0
2026-10-08    0
Name: is_monday_or_tuesday, dtype: int64
```

---

## `cross_sectional` - panel/relative strength features (`quanttro.features.cross_sectional`)

The functions in this module take input in **"wide" format**: index =
timestamp, columns = symbol (ticker), values = that symbol's return/value
at that time. This expresses the same logic as the `add_relative_strength`
and `add_relative_target` functions in `feature_engineering.py` (which
operate on dicts) in a more general, panel-friendly form.

Why "relative"? A market-wide (or sector-wide) common trend - e.g. the
TL's inflation/devaluation trend - is a component shared across ALL BIST
stocks. Using the raw return/target as-is teaches the model a spurious
"always up" signal; using a value RELATIVE to the market (or sector)
removes the spurious correlation created by this common component.

Shared data used in the examples below (returns for the last 5 trading
days):
```python
from quanttro.data import fetch
from quanttro.features import cross_sectional
import pandas as pd

thyao = fetch.fetch_any("THYAO.IS", interval="1d", period="max")
akbnk = fetch.fetch_any("AKBNK.IS", interval="1d", period="max")
sasa = fetch.fetch_any("SASA.IS", interval="1d", period="max")

rets = pd.DataFrame({
    "THYAO.IS": thyao["Close"].pct_change(),
    "AKBNK.IS": akbnk["Close"].pct_change(),
    "SASA.IS": sasa["Close"].pct_change(),
}).dropna().tail(5)
print(rets)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01  0.011474  0.008209  0.066667
2026-10-02  0.019197 -0.006662  0.005208
2026-10-05  0.000856  0.027571  0.005181
2026-10-06 -0.005988  0.007977 -0.020619
2026-10-07 -0.014630 -0.003597 -0.026316
```

### `relative_strength(returns_by_symbol)`

Each stock's relative return against the current market average (across
all symbols) return at that point in time:
`ret_symbol - mean(ret_all_symbols)`, computed row by row. The
generalized form of `add_relative_strength` in `feature_engineering.py`.

**Parameters:**
- `returns_by_symbol` (`pd.DataFrame`): wide format, index=time, columns=symbol

**Returns:** `pd.DataFrame` - same index/columns, each cell shows the
difference from the market average.

**Example:**
```python
relst = cross_sectional.relative_strength(rets)
print(relst)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01 -0.017309 -0.020574  0.037883
2026-10-02  0.013283 -0.012576 -0.000706
2026-10-05 -0.010347  0.016368 -0.006021
2026-10-06  0.000222  0.014187 -0.014409
2026-10-07  0.000218  0.011250 -0.011468
```

### `relative_forward_target(forward_returns_by_symbol)`

The forward (target) returns relative to the market average. The
generalized form of `add_relative_target` in `feature_engineering.py`:
the market-wide trend is subtracted from the target variable, so the
model doesn't learn a spurious "always up" signal. Internally uses the
SAME computation (`_demean_cross_section`) as `relative_strength` - the
only difference is that the input is semantically a forward return.

**Parameters:**
- `forward_returns_by_symbol` (`pd.DataFrame`): wide-format forward returns

**Returns:** `pd.DataFrame` - same index/columns, demeaned.

**Example:**
```python
reltgt = cross_sectional.relative_forward_target(rets)
print(reltgt)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01 -0.017309 -0.020574  0.037883
2026-10-02  0.013283 -0.012576 -0.000706
2026-10-05 -0.010347  0.016368 -0.006021
2026-10-06  0.000222  0.014187 -0.014409
2026-10-07  0.000218  0.011250 -0.011468
```
(In this example, since `rets` is used for both the "return" and the
"forward-return" role, the result is numerically identical to the
`relative_strength` result above - in real usage, `forward_returns_by_symbol`
is typically a shifted (`shift(-horizon)`) target series.)

### `cross_sectional_rank(values_by_symbol)`

At each point in time (row), ranks the symbols by value as a percentile in
`[0, 1]` (the smallest value is close to ~`1/n`, the largest value is
close to `1.0`; tied values get the average rank). `NaN` values are
excluded from that row's ranking.

**Parameters:**
- `values_by_symbol` (`pd.DataFrame`): wide format

**Returns:** `pd.DataFrame` - same index/columns, percentile rank in `[0, 1]`.

**Example:**
```python
csr = cross_sectional.cross_sectional_rank(rets)
print(csr)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01  0.666667  0.333333  1.000000
2026-10-02  1.000000  0.333333  0.666667
2026-10-05  0.333333  1.000000  0.666667
2026-10-06  0.666667  1.000000  0.333333
2026-10-07  0.666667  1.000000  0.333333
```

### `sector_relative_strength(returns_by_symbol, sector_map=None)`

Each stock's relative return against the average return of its OWN SECTOR.

**Parameters:**
- `returns_by_symbol` (`pd.DataFrame`): wide format
- `sector_map` (`dict[str, str] | None`): symbol -> sector name mapping, default `None`
  - If `None`/empty, falls back directly to `relative_strength` (relative
    strength against the whole market).
  - If provided: stocks whose sector is known AND that have at least 2
    symbols in their sector are computed against their own sector average;
    stocks whose sector is unknown OR that are alone in their sector fall
    back to the market average.

**Returns:** `pd.DataFrame` - same index/columns (in the same order) as `returns_by_symbol`.

**Example (without sector_map - falls back to relative_strength):**
```python
srs_nomap = cross_sectional.sector_relative_strength(rets)
print(srs_nomap)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01 -0.017309 -0.020574  0.037883
2026-10-02  0.013283 -0.012576 -0.000706
2026-10-05 -0.010347  0.016368 -0.006021
2026-10-06  0.000222  0.014187 -0.014409
2026-10-07  0.000218  0.011250 -0.011468
```

**Example (with sector_map - THYAO and SASA in the "transport" sector, AKBNK alone in "banking"):**
```python
sector_map = {"THYAO.IS": "transport", "AKBNK.IS": "banking", "SASA.IS": "transport"}
srs = cross_sectional.sector_relative_strength(rets, sector_map)
print(srs)
```

**Output:**
```
            THYAO.IS  AKBNK.IS   SASA.IS
Date                                    
2026-10-01 -0.027596 -0.020574  0.027596
2026-10-02  0.006994 -0.012576 -0.006994
2026-10-05 -0.002163  0.016368  0.002163
2026-10-06  0.007315  0.014187 -0.007315
2026-10-07  0.005843  0.011250 -0.005843
```
(AKBNK falls back to the market average since it is alone in its sector -
its values are identical to the example without `sector_map`. THYAO and
SASA, on the other hand, are computed against their shared "transport"
sector average, so their values have changed.)

---

## High-level shortcut: `bq.make_features`

The `make_features(df, include=("core", "technical", "calendar"))`
function in `quanttro.__init__` automatically combines the most
commonly used functions from `core`/`technical`/`calendar` above for a
single stock:

- `"core"`: `ret_1/5/10/20`, `volume_zscore`, `realized_vol_10`, `atr_14`,
  `at_limit`, `target_fwd_ret` (1-bar forward return target)
- `"technical"`: `rsi_14`, `macd_*`, `bb_*`, `obv`
- `"calendar"`: `day_of_week`/`dow_0..6`, `turn_of_month`, `monday_or_tuesday`

**Note:** This function is for a SINGLE stock only. For multi-stock
(cross-sectional/panel) features (`relative_strength`,
`cross_sectional_rank`, etc.), use the `quanttro.features.cross_sectional`
module directly - these require simultaneous data for multiple stocks, so
they are not included in this function.

**Example:**
```python
import quanttro as bq

df = bq.fetch("THYAO.IS")
feats = bq.make_features(df)
print(feats.shape)
print(feats.columns.tolist())
```

**Output:**
```
(6792, 29)
['close', 'ret_1', 'ret_5', 'ret_10', 'ret_20', 'volume_zscore', 'realized_vol_10', 'atr_14', 'at_limit', 'target_fwd_ret', 'rsi_14', 'macd_macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_%b', 'obv', 'day_of_week', 'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6', 'turn_of_month', 'monday_or_tuesday']
```
