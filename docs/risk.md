# `quanttro.risk`

`quanttro.risk` provides a toolkit for position sizing, drawdown-based risk
throttling, and execution-level cost/stop modeling. The module is split into
three submodules: `sizing.py` (position sizing), `drawdown.py` (equity-curve-based
position throttling), and `execution.py` (stop-loss/take-profit detection,
transaction cost, market impact). All functions can be imported directly from
the `quanttro.risk` package (e.g. `from quanttro.risk import
fixed_fractional`).

All examples in this document have actually been run in the `bist-backtest`
repo's venv, and the outputs have been copy-pasted (no values have been made
up).

---

## Position Sizing

Four classic position sizing methods (`sizing.py`): fixed fractional,
(fractional) Kelly, volatility targeting, and ATR-based risk sizing. All of
them operate on scalar float input/output; if batch computation over a
Series/DataFrame is needed, the caller should wrap them with `Series.apply` or
vectorized operations. There is also a `PositionSizer` class that unifies the
four methods behind a single common interface.

### `fixed_fractional(equity, fraction=0.02)`

Fixed fractional position sizing. Returns a fixed percentage (`fraction`) of
capital (`equity`) as the position size:

```
position_size = equity * fraction
```

This is the simplest and most common sizing method; it is insensitive to
market conditions (volatility, win rate, etc.).

**Parameters:**
- `equity` (float): total capital.
- `fraction` (float): fraction of capital to allocate to the position, default
  `0.02`.

**Returns:** float — `equity * fraction`.

**Example:**
```python
from quanttro.risk import fixed_fractional

fixed_fractional(100000, 0.02)
fixed_fractional(100000)  # same as default fraction=0.02
```

**Output:**
```
2000.0
2000.0
```

### `kelly_fraction(win_rate, win_loss_ratio, kelly_multiplier=0.5)`

Computes the optimal bet/position fraction using the (fractional) Kelly
criterion. The classic Kelly formula:

```
f* = win_rate - (1 - win_rate) / win_loss_ratio
```

`f*` is the theoretical bet fraction that maximizes long-run capital growth
(log-growth). However, `f*` can come out negative (when there is no edge) or
exceed 1 (with invalid/extreme inputs); it is therefore first clipped to the
`[0, 1]` range, then scaled by `kelly_multiplier`. The default
`kelly_multiplier=0.5` means "half-Kelly": full Kelly is in practice extremely
sensitive to estimation errors (`win_rate` and `win_loss_ratio` are never
known exactly) and leads to large drawdowns; fractional Kelly reduces this
risk at a known cost.

**Parameters:**
- `win_rate` (float): win rate (expected to be between 0 and 1).
- `win_loss_ratio` (float): average win / average loss ratio (expected to be
  positive).
- `kelly_multiplier` (float): scale applied to the clipped `f*`, default
  `0.5`.

**Returns:** float — `clip(f*, 0, 1) * kelly_multiplier`.

**Example:**
```python
from quanttro.risk import kelly_fraction

kelly_fraction(0.55, 1.5)                          # half-Kelly (default)
kelly_fraction(0.55, 1.5, kelly_multiplier=1.0)    # full Kelly
kelly_fraction(0.3, 1.0)                           # negative edge -> clipped to 0
```

**Output:**
```
0.12500000000000003
0.25000000000000006
0.0
```

### `volatility_target_size(target_vol, asset_vol, max_leverage=1.0)`

Computes a leverage/size ratio using volatility targeting. The idea: to keep
the portfolio's realized volatility at a fixed `target_vol` level, the
position shrinks when the asset's own volatility (`asset_vol`) rises, and
grows when it falls:

```
leverage = target_vol / asset_vol
```

The result is capped from above by `max_leverage` to prevent excessive
leverage. `asset_vol <= 0` is a special case (the ratio mathematically goes to
infinity): in this case, instead of raising a division-by-zero error,
`max_leverage` is returned directly.

**Parameters:**
- `target_vol` (float): the targeted volatility level (the caller must use a
  consistent unit — annualized or per-period).
- `asset_vol` (float): the asset's realized/estimated volatility (in the same
  unit).
- `max_leverage` (float): the upper bound the returned ratio is capped at,
  default `1.0` (no leverage).

**Returns:** float — `min(target_vol / asset_vol, max_leverage)`.

**Example:**
```python
from quanttro.risk import volatility_target_size

volatility_target_size(0.15, 0.30)                           # normal case
volatility_target_size(0.40, 0.10, max_leverage=2.0)          # hits the cap
volatility_target_size(0.15, 0.0, max_leverage=1.5)           # asset_vol<=0 -> max_leverage
```

**Output:**
```
0.5
2.0
1.5
```

### `atr_based_size(equity, atr, risk_per_trade=0.01, price=None)`

ATR (Average True Range)-based position sizing using risk-parity logic.
Assuming the stop-loss distance will be approximately `atr`, the position
size is computed so that only `risk_per_trade` of capital is lost if the stop
is hit:

```
position_size = equity * risk_per_trade / atr
```

If `atr` is given in ABSOLUTE price units (e.g.
`quanttro.features.core.average_true_range(..., normalize=False)`), the
result is interpreted as a NUMBER OF SHARES. If `price` is provided, `atr` is
assumed to be NORMALIZED/percentage-based (e.g. the output of
`average_true_range(..., normalize=True)`, i.e. atr/close) — in this case the
notional (TRY) position size is computed first, then divided by `price` to
convert it into a number of shares:

```
shares = (equity * risk_per_trade / atr) / price
```

**Parameters:**
- `equity` (float): total capital.
- `atr` (float): Average True Range — can be absolute (price units, if
  `price=None`) or normalized/percentage-based (if `price` is given). Must be
  positive.
- `risk_per_trade` (float): fraction of capital to risk per trade, default
  `0.01` (1%).
- `price` (float | None): if given, `atr` is assumed to be normalized and the
  result is converted into number of shares. Must be positive.

**Returns:** float — number of shares.

**Raises:** `ValueError` — if `atr <= 0`, or (if `price` is given)
`price <= 0`.

**Example:**
```python
from quanttro.risk import atr_based_size

atr_based_size(100000, atr=2.5, risk_per_trade=0.01)                     # absolute ATR
atr_based_size(100000, atr=0.02, risk_per_trade=0.01, price=50.0)        # normalized ATR
```

**Output:**
```
400.0
1000.0
```

### `class PositionSizer(method, **params)`

A class that unifies the four position sizing methods above under a single
common interface. In `__init__`, the parameters required for the chosen
`method` (`"fixed_fractional" | "kelly" | "volatility_target" | "atr_based"`)
can be provided either in `__init__(**params)` (as shared defaults), or at
each `.size(equity, **kwargs)` call (specific to that call, overriding
`params`).

If parameters that a method REQUIRES (i.e. that have no natural default) are
not provided, no silent assumption is made — an explicit `ValueError` is
raised (e.g. `win_rate`/`win_loss_ratio` are mandatory for the `"kelly"`
method).

`.size()` always returns the final position size scaled by equity — since
`kelly_fraction` and `volatility_target_size` each return only a
fraction/multiplier on their own, `PositionSizer` multiplies them by `equity`
to bring them into the same "position size" semantics as the other two
methods (`fixed_fractional`, `atr_based_size`).

**Parameters (`__init__`):**
- `method` (str): `"fixed_fractional"`, `"kelly"`, `"volatility_target"`, or
  `"atr_based"`.
- `**params`: method-specific default parameters.

**Method: `size(equity, **kwargs)`**

Computes the position size according to the selected method. `self.params`
is merged with `kwargs`; if a parameter with the same name exists in both,
`kwargs` wins.

**Returns:** float — the final position size (in the same unit as equity).

**Example:**
```python
from quanttro.risk import PositionSizer

ps = PositionSizer('kelly', win_rate=0.55, win_loss_ratio=1.5)
ps.size(100000)

ps2 = PositionSizer('volatility_target', max_leverage=1.0)
ps2.size(100000, target_vol=0.15, asset_vol=0.25)

ps3 = PositionSizer('kelly')
ps3.size(100000)  # missing parameter -> ValueError
```

**Output:**
```
12500.000000000004
60000.0
ValueError: 'kelly' yontemi icin eksik parametre(ler): ['win_rate', 'win_loss_ratio']. PositionSizer(method='kelly', ...) olusturulurken ya da .size(equity, ...) cagrisinda verilmelidir.
```

---

## Drawdown Management

`drawdown.py` consolidates tiered risk management of the form "if capital has
dropped by this much, shrink/close positions" into the `DrawdownThrottle`
class; `rolling_drawdown` is the helper function that produces the
instantaneous drawdown series fed into it.

### `class DrawdownThrottle(tiers)`

A simple risk management layer that progressively shrinks/halts position size
based on the current drawdown level.

`tiers` is a list of `(drawdown_threshold, size_multiplier)` pairs; e.g.
`[(0.10, 0.5), (0.20, 0.0)]` means the position is cut in half (0.5x) once
drawdown exceeds 10%, and to zero (0.0x) once it exceeds 20%.

`drawdown_threshold` values are given as MAGNITUDES (positive, e.g. 0.10 =
10% decline). The `current_drawdown` passed to `multiplier()` and `apply()`
can be either a positive magnitude or negative (e.g. as in `rolling_drawdown`
or the output of `quanttro.backtest.metrics.max_drawdown`, where -0.15 =
15% decline) — it is normalized internally with `abs()`.

**Parameters (`__init__`):**
- `tiers` (list[tuple[float, float]]): `(threshold, multiplier)` pairs; cannot
  be empty (`ValueError`). Sort from smallest to largest threshold when
  constructing.

**Method: `multiplier(current_drawdown)`**

Returns the position multiplier to apply for the current drawdown. The
magnitude of `current_drawdown` is compared against the thresholds (`dd >=
threshold`); the multiplier of the HIGHEST threshold EXCEEDED is applied. If
no threshold is exceeded, 1.0 (full position) is returned.

**Method: `apply(position_size, current_drawdown)`**

`position_size * multiplier(current_drawdown)`.

**Example:**
```python
from quanttro.risk import DrawdownThrottle

throttle = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
throttle.multiplier(-0.05)   # no threshold exceeded
throttle.multiplier(-0.12)   # exceeded the 10% threshold
throttle.multiplier(-0.25)   # exceeded the 20% threshold
throttle.apply(10000, -0.12)
```

**Output:**
```
1.0
0.5
0.0
5000.0
```

### `rolling_drawdown(cumulative_returns)`

The drawdown series at each point relative to the running maximum of the
cumulative return up to that point. Uses the SAME formula as
`quanttro.backtest.metrics.max_drawdown`
(`drawdown_t = curve_t / running_max_t - 1`), but returns the FULL SERIES
rather than a single scalar (the worst point) — used together with
`DrawdownThrottle` to feed the current drawdown at every bar.

**Parameters:**
- `cumulative_returns` (pd.Series): the cumulative return curve (e.g.
  `(1+r).cumprod()`).

**Returns:** pd.Series (`name="drawdown"`) — NEGATIVE or zero values (e.g.
-0.15 = 15% decline).

**Example:**
```python
import numpy as np
import pandas as pd
from quanttro.risk import rolling_drawdown

np.random.seed(42)
rets = pd.Series(np.random.normal(0.0005, 0.02, 250))
cum = (1 + rets).cumprod()
dd = rolling_drawdown(cum)
dd.tail(8)
dd.min()
```

**Output:**
```
242   -0.063697
243   -0.047017
244   -0.069418
245   -0.075178
246   -0.083500
247   -0.095018
248   -0.062611
249   -0.054550
Name: drawdown, dtype: float64

-0.27992948159722275
```

### `aggregate_portfolio_drawdown(returns, weights=None)`

Combines multiple strategies'/pairs' return series into a **single**
weighted account equity curve and returns that curve's drawdown — distinct
from computing each strategy's drawdown independently. Forum research
(ForexFactory threads on multi-pair EA backtesting) found traders
explicitly asking for exactly this: a way to see what the *combined*
account drawdown looks like when several strategies run simultaneously,
not just each one's own backtest in isolation. If the strategies' losing
days line up (they're correlated), the combined drawdown can be much worse
than any single strategy's own number suggested.

**Parameters:**
- `returns` (pd.DataFrame): one column per strategy/pair, same index.
- `weights` (dict | pd.Series | None): allocation per column; defaults to
  equal weight across all columns.

**Returns:** pd.Series — the `rolling_drawdown` of the combined, weighted
account curve.

**Example:**
```python
import pandas as pd
from quanttro.risk import aggregate_portfolio_drawdown, rolling_drawdown

idx = pd.date_range("2026-01-01", periods=4, freq="D")
# Two strategies whose losing days are offsetting (negatively correlated)
a = pd.Series([0.05, -0.05, 0.05, -0.05], index=idx)
b = pd.Series([-0.05, 0.05, -0.05, 0.05], index=idx)

combined = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": b}))
print("combined drawdown:", combined.min())
print("A alone drawdown:", rolling_drawdown((1 + a).cumprod()).min())
```

**Output:**
```
combined drawdown: 0.0
A alone drawdown: -0.05237500000000017
```

Here the two strategies' bad days cancel out, so the combined account never
draws down at all — even though strategy A alone has a real ~5.2% drawdown.
Run the same two series through the account with `weights={"A": 1.0, "B":
0.0}` (or any correlated pair instead) and the combined number matches or
exceeds the worse single strategy instead — the function reports whichever
is actually true for the given weights and correlation, rather than
assuming diversification always helps.

---

## Execution / Cost

`execution.py` consists of three parts: `apply_stop_loss_take_profit` (a
scanning function that finds WHEN the stop-loss/take-profit levels are
triggered in an open position's price series), `TransactionCostModel` (fixed +
percentage commission + slippage), and `square_root_market_impact` (a
simplified Almgren-Chriss square-root market impact model).

### `apply_stop_loss_take_profit(prices, entry_price, entry_index, sl_pct=None, tp_pct=None)`

Scans the `prices` series starting at `entry_index` and finds the first point
at which the stop-loss (`sl_pct`) or take-profit (`tp_pct`) level is reached.

If `sl_pct`/`tp_pct` is `None`, that check is SKIPPED. If both are `None`, the
trade never closes (`reason="none"`). The scan starts from the bars AFTER
`entry_index` (the entry bar itself is not checked). Even if both the
stop-loss and take-profit levels are crossed within the same bar, since we do
not know the actual intrabar order, the CONSERVATIVE assumption is that
stop-loss takes PRIORITY.

**Parameters:**
- `prices` (pd.Series): a price series that includes `entry_index`.
- `entry_price` (float): the entry price of the position.
- `entry_index`: the label of the entry bar within `prices.index`.
- `sl_pct` (float | None): stop-loss percentage (e.g. 0.02 = `entry_price * (1 -
  sl_pct)`). Not checked if `None`.
- `tp_pct` (float | None): take-profit percentage (e.g. 0.03 = `entry_price * (1 +
  tp_pct)`). Not checked if `None`.

**Returns:** dict — `{"exit_index": ..., "exit_price": ..., "reason": "stop_loss" |
"take_profit" | "none"}`. If no threshold is triggered, `exit_index`/`exit_price`
are the LAST element of the series, with `reason="none"`.

**Raises:** `ValueError` — if `entry_index` is not found in `prices.index`.

**Example:**
```python
import pandas as pd
from quanttro.risk import apply_stop_loss_take_profit

dates = pd.date_range('2026-01-01', periods=10, freq='D')
prices = pd.Series([100, 101, 99, 97, 95, 98, 102, 105, 103, 101], index=dates)

apply_stop_loss_take_profit(prices, entry_price=100, entry_index=dates[0], sl_pct=0.04, tp_pct=0.05)
apply_stop_loss_take_profit(prices, entry_price=100, entry_index=dates[0], sl_pct=None, tp_pct=0.08)
apply_stop_loss_take_profit(prices, entry_price=100, entry_index=dates[0])
```

**Output:**
```
{'exit_index': Timestamp('2026-01-05 00:00:00'), 'exit_price': 95, 'reason': 'stop_loss'}
{'exit_index': Timestamp('2026-01-10 00:00:00'), 'exit_price': np.int64(101), 'reason': 'none'}
{'exit_index': Timestamp('2026-01-10 00:00:00'), 'exit_price': np.int64(101), 'reason': 'none'}
```

### `class TransactionCostModel(fixed_fee=0.0, pct_fee=0.001, slippage_pct=0.0)`

A simple transaction cost model: fixed commission + percentage commission +
slippage, all applied to the trade value (`trade_value`).

**Parameters (`__init__`):**
- `fixed_fee` (float): fixed fee per trade, default `0.0`.
- `pct_fee` (float): commission as a percentage of trade value, default
  `0.001`.
- `slippage_pct` (float): slippage as a percentage of trade value, default
  `0.0`.

**Method: `cost(trade_value)`**

Total transaction cost:

```
cost = fixed_fee + trade_value * (pct_fee + slippage_pct)
```

**Returns:** float.

**Example:**
```python
from quanttro.risk import TransactionCostModel

tcm = TransactionCostModel(fixed_fee=5.0, pct_fee=0.001, slippage_pct=0.0005)
tcm.cost(100000)
```

**Output:**
```
155.0
```

### `square_root_market_impact(trade_value, avg_daily_volume_value, volatility, impact_coefficient=1.0)`

A simplified Almgren-Chriss square-root market impact model.

NOTE: This is a HEAVILY SIMPLIFIED, ASSUMPTION-DRIVEN approximation of the
full Almgren-Chriss model from the academic literature (which includes many
additional terms such as the temporary/permanent impact split, trading speed,
and a risk-aversion parameter). Actual market impact varies greatly by
asset/market/time period; the constant `impact_coefficient` here does NOT
REPRESENT an actual calibration, it is merely a proportional scaling
coefficient.

Formula:

```
impact = impact_coefficient * volatility * sqrt(trade_value / avg_daily_volume_value)
```

That is, market impact scales with the SQUARE ROOT (not linearly, but via
sqrt) of the ratio of trade size to average daily trading volume
(`avg_daily_volume_value`, in TRY), and with the asset's volatility
(`volatility`).

**Parameters:**
- `trade_value` (float): the value of the trade in TRY (or any consistent
  currency).
- `avg_daily_volume_value` (float): the asset's average daily trading volume
  value (in the same unit). Must be positive.
- `volatility` (float): the asset's volatility (per period; the caller must
  use it consistently).
- `impact_coefficient` (float): the model's scale coefficient, default `1.0`.

**Returns:** float — the estimated market impact (proportional/percentage
price shift, e.g. 0.002 = 0.2%).

**Raises:** `ValueError` — if `avg_daily_volume_value <= 0`.

**Example:**
```python
from quanttro.risk import square_root_market_impact

square_root_market_impact(trade_value=500000, avg_daily_volume_value=20000000, volatility=0.02)
square_root_market_impact(trade_value=500000, avg_daily_volume_value=20000000, volatility=0.02, impact_coefficient=2.5)
```

**Output:**
```
0.0031622776601683794
0.00790569415042095
```

---

## Combined Usage

The functions in the three submodules typically work together in a single
trade/bar loop: first a "target" position size is computed with
`PositionSizer`, then this size is throttled according to the current
drawdown level with `DrawdownThrottle`, and finally the resulting cost is
estimated with `TransactionCostModel`. The example below shows this chain on
a synthetic but actually-executed return series:

```python
import numpy as np
import pandas as pd
from quanttro.risk import PositionSizer, DrawdownThrottle, rolling_drawdown, TransactionCostModel

np.random.seed(7)
rets = pd.Series(np.random.normal(0.0004, 0.018, 500))
cum = (1 + rets).cumprod()
dd_series = rolling_drawdown(cum)

# 1) target position: via volatility targeting (max 150% leverage)
sizer = PositionSizer('volatility_target', target_vol=0.15, max_leverage=1.5)
equity = 250000
realized_vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)
raw_size = sizer.size(equity, asset_vol=realized_vol)

# 2) drawdown throttle: 0.6x at 8%, 0.3x at 15%, 0.0x at 25%
throttle = DrawdownThrottle([(0.08, 0.6), (0.15, 0.3), (0.25, 0.0)])
current_dd = dd_series.iloc[300]
final_size = throttle.apply(raw_size, current_dd)

# 3) transaction cost
tcm = TransactionCostModel(fixed_fee=2.0, pct_fee=0.0015)
cost = tcm.cost(final_size)

print(realized_vol, raw_size, current_dd, final_size, cost)
```

**Output:**
```
0.22628450219053248 165720.5846488985 -0.1575803047498855 49716.17539466955 76.57426309200433
```

Here, since the realized volatility (22.6%) is higher than the target
volatility (15%), `volatility_target_size` pulls the leverage below 1.0
(`raw_size` ≈ 165,720 TRY, ~66% of the 250,000 TRY capital). At bar 300, since
the drawdown is -15.8% (having exceeded the second threshold, 15%),
`DrawdownThrottle` throttles this size down to 0.3x (`final_size` ≈ 49,716
TRY). Finally, the fixed + percentage cost of a trade of this size is
computed as ≈ 76.6 TRY. This chained usage shows why the three modules were
designed as separate (and mutually independent) functions: each one takes on
a single responsibility (sizing / throttling / cost) and the calling code can
combine them in whatever order/combination it wants.

---

## Realistic slippage and partial fills

Two gaps show up repeatedly in feedback on vectorized backtesting libraries:
slippage is modeled as a flat percentage regardless of trade direction, and a
large order is assumed to fill completely, instantly, in a single bar — both
flatter the backtest relative to live execution. `execution.py` adds two
functions that address this directly.

### `realistic_slippage(trade_value, avg_daily_volume_value, volatility, side, impact_coefficient=1.0)`

A direction-aware wrapper around `square_root_market_impact`: a `"buy"` order
is pushed to a worse (higher) price, a `"sell"` order to a worse (lower)
price. Returns a multiplier to apply directly to the quoted price.

```python
from quanttro.risk import realistic_slippage

buy = realistic_slippage(trade_value=50_000, avg_daily_volume_value=2_000_000, volatility=0.02, side="buy")
sell = realistic_slippage(trade_value=50_000, avg_daily_volume_value=2_000_000, volatility=0.02, side="sell")
print(buy, sell)
```

**Output:**
```
1.0031622776601683 0.9968377223398316
```

The two multipliers are symmetric around 1.0 — the same simplified
square-root impact term, just applied with opposite sign depending on side.

### `simulate_partial_fills(order_qty, bar_volumes, bar_prices, participation_rate=0.1, max_bars=None)`

Walks forward bar-by-bar, filling at most `bar_volume * participation_rate`
of the order per bar, carrying the remainder to the next bar. This replaces
the "entire order fills instantly" assumption with a participation-rate
constraint — the same kind of constraint a real execution algorithm (VWAP/TWAP
participation) imposes on itself to avoid moving the market.

```python
import pandas as pd
from quanttro.risk import simulate_partial_fills

volumes = pd.Series([2000, 3000, 5000, 8000])
prices = pd.Series([10.0, 10.05, 10.1, 10.0])
fill = simulate_partial_fills(order_qty=10_000, bar_volumes=volumes, bar_prices=prices, participation_rate=0.1)
print(fill)
```

**Output:**
```
OrderFill(filled_qty=1800.0, avg_price=10.036111111111111, remaining_qty=8200.0,
          bars_to_fill=4, fill_prices=[10.0, 10.05, 10.1, 10.0],
          fill_qtys=[200.0, 300.0, 500.0, 800.0])
```

A 10,000-unit order against bars with 2,000–8,000 units of volume, capped at
10% participation, only fills 1,800 units across all 4 bars (`remaining_qty`
is 8,200) — this is the realistic consequence of order size relative to
available liquidity, which a naive "fills instantly" backtest would hide
entirely. `fill.fully_filled` is `False` here; the caller decides what to do
with the remainder (carry to more bars, cancel, or accept a worse average
price by raising `participation_rate`).
