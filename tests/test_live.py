"""quanttro.live paketi icin birim testleri.

pytest ile calistirilir::

    python -m pytest tests/test_live.py -v
"""
from __future__ import annotations

import pytest

from quanttro.live.broker import Order, OrderSide, OrderType, Position, SimulatedBroker
from quanttro.live.loop import LiveTradingLoop
from quanttro.risk.execution import TransactionCostModel


def make_price_feed(prices: dict[str, float]):
    return lambda symbol: prices[symbol]


# --------------------------------------------------------------------------
# SimulatedBroker
# --------------------------------------------------------------------------


def test_simulated_broker_buy_updates_position_and_cash():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    order = broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))

    assert order.status == "filled"
    assert order.filled_qty == 10
    assert order.avg_fill_price == pytest.approx(100.0)
    assert broker.get_cash() == pytest.approx(10_000.0 - 1000.0)
    assert broker.get_positions()["THYAO.IS"].qty == 10


def test_simulated_broker_applies_transaction_cost():
    cost_model = TransactionCostModel(fixed_fee=5.0, pct_fee=0.001)
    broker = SimulatedBroker(
        make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0, cost_model=cost_model
    )
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    expected_cash = 10_000.0 - 1000.0 - cost_model.cost(1000.0)
    assert broker.get_cash() == pytest.approx(expected_cash)


def test_simulated_broker_sell_reduces_position():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.SELL, qty=4))
    assert broker.get_positions()["THYAO.IS"].qty == pytest.approx(6.0)


def test_simulated_broker_sell_to_zero_closes_position():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.SELL, qty=10))
    assert "THYAO.IS" not in broker.get_positions()


def test_simulated_broker_rejects_order_exceeding_cash():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=50.0)
    order = broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    assert order.status == "cancelled"
    assert broker.get_positions() == {}
    assert broker.get_cash() == pytest.approx(50.0)


def test_simulated_broker_weighted_avg_price_on_additional_buy():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    broker.price_feed = make_price_feed({"THYAO.IS": 200.0})
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    # (10*100 + 10*200) / 20 = 150
    assert broker.get_positions()["THYAO.IS"].avg_price == pytest.approx(150.0)


def test_simulated_broker_account_value_uses_mark_prices():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    value_at_cost = broker.get_account_value()
    value_marked = broker.get_account_value(mark_prices={"THYAO.IS": 150.0})
    assert value_marked > value_at_cost


def test_simulated_broker_cancel_order_returns_false():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    assert broker.cancel_order(order_id=1) is False


def test_simulated_broker_order_history_tracks_all_orders():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=10_000.0)
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.BUY, qty=10))
    broker.place_order(Order(symbol="THYAO.IS", side=OrderSide.SELL, qty=5))
    assert len(broker.order_history) == 2


# --------------------------------------------------------------------------
# LiveTradingLoop
# --------------------------------------------------------------------------


def test_live_trading_loop_opens_position_toward_target():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    loop = LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 50})
    orders = loop.step()

    assert len(orders) == 1
    assert orders[0].side == OrderSide.BUY
    assert orders[0].qty == pytest.approx(50.0)
    assert broker.get_positions()["THYAO.IS"].qty == pytest.approx(50.0)


def test_live_trading_loop_reduces_toward_lower_target():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    loop = LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 50})
    loop.step()

    loop2 = LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 20})
    orders = loop2.step()

    assert orders[0].side == OrderSide.SELL
    assert orders[0].qty == pytest.approx(30.0)
    assert broker.get_positions()["THYAO.IS"].qty == pytest.approx(20.0)


def test_live_trading_loop_closes_position_when_target_drops_out():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 50}).step()

    loop = LiveTradingLoop(broker, signal_fn=lambda: {})
    orders = loop.step()

    assert orders[0].side == OrderSide.SELL
    assert "THYAO.IS" not in broker.get_positions()


def test_live_trading_loop_skips_trades_below_min_qty():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 50}).step()

    loop = LiveTradingLoop(broker, signal_fn=lambda: {"THYAO.IS": 50 + 1e-9}, min_trade_qty=1e-6)
    orders = loop.step()
    assert orders == []


def test_live_trading_loop_passes_through_signal_args():
    broker = SimulatedBroker(make_price_feed({"THYAO.IS": 100.0}), initial_cash=100_000.0)
    seen = {}

    def signal_fn(timestamp):
        seen["timestamp"] = timestamp
        return {"THYAO.IS": 10}

    loop = LiveTradingLoop(broker, signal_fn)
    loop.step("2026-10-10")
    assert seen["timestamp"] == "2026-10-10"


def test_order_and_position_dataclasses_basic_fields():
    order = Order(symbol="AKBNK.IS", side=OrderSide.BUY, qty=5, order_type=OrderType.LIMIT, limit_price=50.0)
    assert order.status == "pending"
    assert order.order_id > 0

    position = Position(symbol="AKBNK.IS", qty=5, avg_price=50.0)
    assert position.market_value == pytest.approx(250.0)
