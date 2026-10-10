# `quanttro.live`

## Scope — read this before anything else

**This module does not ship a connection to any real BIST brokerage.** A real
integration depends on a specific broker's REST/FIX API, an API key, and a
regulatory/compliance process — all brokerage-specific, and outside what a
general-purpose library can responsibly bundle. What `quanttro.live`
actually provides is the *shape* of that integration: a broker-agnostic
interface (`BrokerInterface`) so that a strategy's signal-generation code
never has to change when moving from paper trading to a real broker — only
the broker implementation underneath it does.

Three pieces, in `broker.py` and `loop.py`:

- `Order`, `OrderSide`, `OrderType`, `Position` — plain data representations.
- `BrokerInterface` (abstract) — `place_order`, `cancel_order`,
  `get_positions`, `get_cash`, `get_account_value`. To connect a real broker,
  subclass this and implement these five methods against that broker's API.
- `SimulatedBroker(BrokerInterface)` — a paper-trading implementation that
  uses `quanttro.risk.execution`'s cost/slippage models for realistic fills.
- `LiveTradingLoop` — turns "target positions" (from a signal function) into
  the buy/sell orders needed to get there, via whatever `BrokerInterface` it's
  given.

All examples below were actually run against this repo's venv.

---

## `SimulatedBroker`

```python
from quanttro.live import SimulatedBroker
from quanttro.risk import TransactionCostModel

price_feed = lambda sym: {"THYAO.IS": 300.0, "AKBNK.IS": 55.0}[sym]
broker = SimulatedBroker(
    price_feed,
    initial_cash=1_000_000,
    cost_model=TransactionCostModel(fixed_fee=0.0, pct_fee=0.001),
)
```

`price_feed` is any callable `symbol -> price` — a closure over the last
known close in a backtest, or a closure reading from a live market-data feed
in production; `SimulatedBroker` doesn't care which. `cost_model` is optional;
passing `None` means zero-cost fills (fine for quick prototyping, not for a
result you intend to trust).

`place_order` fills immediately at the current `price_feed` price (optionally
shifted by `realistic_slippage` if `market_impact_kwargs` is supplied),
deducts `cost_model.cost(...)`, and updates `cash`/`positions`:

```python
from quanttro.live import Order, OrderSide

order = broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=200))
print(order)
print(broker.get_cash(), broker.get_positions())
```

**Output:**
```
Order(symbol='THYAO.IS', side=<OrderSide.BUY: 'buy'>, qty=200.0, order_type=<OrderType.MARKET: 'market'>,
      limit_price=None, order_id=1, status='filled', filled_qty=200.0, avg_fill_price=300.0)
939940.0 {'THYAO.IS': Position(symbol='THYAO.IS', qty=200.0, avg_price=300.0)}
```

An order that would require more cash than is available is rejected
(`status="cancelled"`) rather than silently allowing negative cash — this
matters because a backtest that lets a strategy go arbitrarily negative on
cash will show returns no real account could have produced.

---

## `LiveTradingLoop`

A strategy rarely emits individual buy/sell orders directly — it emits
*target positions* ("I want 200 THYAO.IS and 1000 AKBNK.IS right now").
`LiveTradingLoop.step()` diffs that target against the broker's current
positions and places exactly the orders needed to close the gap:

```python
from quanttro.live import LiveTradingLoop

def signal_fn():
    return {"THYAO.IS": 200, "AKBNK.IS": 1000}

loop = LiveTradingLoop(broker, signal_fn)
orders = loop.step()
for o in orders:
    print(o)
print("account value:", broker.get_account_value())
```

**Output:**
```
Order(symbol='AKBNK.IS', side=<OrderSide.BUY: 'buy'>, qty=1000.0, ..., status='filled', avg_fill_price=55.0)
account value: 999885.0
```

(THYAO.IS was already at its target from the previous order, so only AKBNK.IS
generated a trade.) On the next call, if the target changes, only the *delta*
is traded:

```python
def signal_fn2():
    return {"THYAO.IS": 50, "AKBNK.IS": 1000}

loop2 = LiveTradingLoop(broker, signal_fn2)
for o in loop2.step():
    print(o)
print(broker.get_positions())
```

**Output:**
```
Order(symbol='THYAO.IS', side=<OrderSide.SELL: 'sell'>, qty=150.0, ..., status='filled', avg_fill_price=300.0)
{'THYAO.IS': Position(symbol='THYAO.IS', qty=50.0, avg_price=300.0), 'AKBNK.IS': Position(symbol='AKBNK.IS', qty=1000.0, avg_price=55.0)}
```

`LiveTradingLoop` deliberately does not decide *when* `step()` is called —
that's a scheduling concern (cron, a websocket bar event, exchange hours)
that differs between backtest and live contexts; `step()` only answers "given
the current target, what trades are needed right now."

---

## Connecting a real broker

Subclass `BrokerInterface` and implement its five abstract/overridden
methods against the target broker's API (place a real order, cancel a real
order, fetch real positions, fetch real cash balance). Nothing else in this
module — `Order`/`Position`/`LiveTradingLoop` — needs to change; a strategy
written against `BrokerInterface` runs unmodified against `SimulatedBroker`
in testing and the real subclass in production. This library does not ship
that subclass for any specific BIST brokerage, and doing so correctly also
requires handling that broker's authentication, rate limits, and order-status
polling/webhooks — all out of scope here.
