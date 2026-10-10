"""quanttro.risk paketi icin birim testleri.

Her fonksiyon/sinif, formulun bagimsiz (el ile ya da numpy ile) hesaplanan
degeriyle karsilastirilarak ya da bilinen sinir durumlariyla dogrulanir.
pytest ile calistirilir::

    python -m pytest tests/test_risk.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttro.backtest.metrics import max_drawdown
from quanttro.risk.drawdown import DrawdownThrottle, aggregate_portfolio_drawdown, rolling_drawdown
from quanttro.risk.execution import (
    OrderFill,
    TransactionCostModel,
    apply_stop_loss_take_profit,
    realistic_slippage,
    simulate_partial_fills,
    square_root_market_impact,
)
from quanttro.risk.sizing import (
    PositionSizer,
    atr_based_size,
    fixed_fractional,
    kelly_fraction,
    volatility_target_size,
)

# --------------------------------------------------------------------------
# sizing.py
# --------------------------------------------------------------------------


def test_fixed_fractional_returns_percentage_of_equity():
    assert fixed_fractional(100_000, fraction=0.02) == pytest.approx(2_000)


def test_fixed_fractional_uses_default_fraction_of_two_percent():
    assert fixed_fractional(100_000) == pytest.approx(2_000)


def test_kelly_fraction_matches_formula_with_half_kelly_default():
    # f* = 0.6 - 0.4/2.0 = 0.4; varsayilan kelly_multiplier=0.5 -> 0.2
    win_rate, win_loss_ratio = 0.6, 2.0
    f_star = win_rate - (1 - win_rate) / win_loss_ratio
    assert kelly_fraction(win_rate, win_loss_ratio) == pytest.approx(f_star * 0.5)


def test_kelly_fraction_full_kelly_matches_raw_formula():
    win_rate, win_loss_ratio = 0.6, 2.0
    f_star = win_rate - (1 - win_rate) / win_loss_ratio
    assert kelly_fraction(win_rate, win_loss_ratio, kelly_multiplier=1.0) == pytest.approx(f_star)


def test_kelly_fraction_clips_negative_edge_to_zero():
    # win_rate=0.3, win_loss_ratio=1.0 -> f* = 0.3 - 0.7/1.0 = -0.4 (negatif
    # edge: "bahis yapma" anlaminda, 0'a kirpilmali).
    assert kelly_fraction(0.3, 1.0, kelly_multiplier=1.0) == pytest.approx(0.0)


def test_kelly_fraction_clips_above_one_for_out_of_domain_win_rate():
    # win_rate=1.5 gecersiz/hatali bir girdidir (olasilik [0,1] disinda);
    # f* = 1.5 - (1-1.5)/2.0 = 1.75 -> clip ile 1.0'a, sonra kelly_multiplier=1.0
    # ile 1.0 olarak kalmali (savunmaci kirpma davranisi).
    assert kelly_fraction(1.5, 2.0, kelly_multiplier=1.0) == pytest.approx(1.0)


def test_volatility_target_size_matches_ratio_when_under_max_leverage():
    assert volatility_target_size(0.15, 0.30, max_leverage=1.0) == pytest.approx(0.5)


def test_volatility_target_size_caps_at_max_leverage():
    # ham oran 0.30/0.10 = 3.0, max_leverage=2.0 ile sinirlanmali
    assert volatility_target_size(0.30, 0.10, max_leverage=2.0) == pytest.approx(2.0)


def test_volatility_target_size_zero_asset_vol_returns_max_leverage():
    assert volatility_target_size(0.1, 0.0, max_leverage=1.5) == pytest.approx(1.5)


def test_atr_based_size_matches_formula_without_price():
    equity, atr, risk_per_trade = 100_000, 2.0, 0.01
    expected = equity * risk_per_trade / atr
    assert atr_based_size(equity, atr, risk_per_trade=risk_per_trade) == pytest.approx(expected)


def test_atr_based_size_converts_notional_to_shares_with_price():
    equity, atr, risk_per_trade, price = 100_000, 0.02, 0.01, 50.0
    notional = equity * risk_per_trade / atr
    expected_shares = notional / price
    assert atr_based_size(equity, atr, risk_per_trade=risk_per_trade, price=price) == pytest.approx(
        expected_shares
    )


def test_atr_based_size_rejects_non_positive_atr():
    with pytest.raises(ValueError):
        atr_based_size(100_000, atr=0.0)


def test_atr_based_size_rejects_non_positive_price():
    with pytest.raises(ValueError):
        atr_based_size(100_000, atr=1.0, price=0.0)


# --------------------------------------------------------------------------
# sizing.py - PositionSizer
# --------------------------------------------------------------------------


def test_position_sizer_fixed_fractional_matches_function():
    sizer = PositionSizer("fixed_fractional", fraction=0.05)
    assert sizer.size(100_000) == pytest.approx(fixed_fractional(100_000, fraction=0.05))


def test_position_sizer_kelly_scales_fraction_by_equity():
    sizer = PositionSizer("kelly", win_rate=0.6, win_loss_ratio=2.0)
    expected = 100_000 * kelly_fraction(0.6, 2.0)
    assert sizer.size(100_000) == pytest.approx(expected)


def test_position_sizer_kelly_missing_required_param_raises_value_error():
    sizer = PositionSizer("kelly", win_rate=0.6)  # win_loss_ratio eksik
    with pytest.raises(ValueError):
        sizer.size(100_000)


def test_position_sizer_volatility_target_scales_leverage_by_equity():
    sizer = PositionSizer("volatility_target", target_vol=0.15, asset_vol=0.30)
    assert sizer.size(100_000) == pytest.approx(100_000 * 0.5)


def test_position_sizer_atr_based_kwargs_override_init_params():
    sizer = PositionSizer("atr_based", atr=2.0, risk_per_trade=0.01)
    expected = atr_based_size(100_000, atr=2.0, risk_per_trade=0.02)
    assert sizer.size(100_000, risk_per_trade=0.02) == pytest.approx(expected)


def test_position_sizer_atr_based_missing_atr_raises_value_error():
    sizer = PositionSizer("atr_based")
    with pytest.raises(ValueError):
        sizer.size(100_000)


def test_position_sizer_rejects_unknown_method():
    with pytest.raises(ValueError):
        PositionSizer("not_a_method")


# --------------------------------------------------------------------------
# drawdown.py
# --------------------------------------------------------------------------


def test_drawdown_throttle_no_tier_crossed_returns_full_size():
    throttle = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
    assert throttle.multiplier(0.05) == pytest.approx(1.0)


def test_drawdown_throttle_selects_highest_crossed_tier():
    throttle = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
    assert throttle.multiplier(0.15) == pytest.approx(0.5)
    assert throttle.multiplier(0.25) == pytest.approx(0.0)


def test_drawdown_throttle_tier_order_in_constructor_does_not_matter():
    throttle_sorted = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
    throttle_unsorted = DrawdownThrottle([(0.20, 0.0), (0.10, 0.5)])
    assert throttle_sorted.multiplier(0.15) == throttle_unsorted.multiplier(0.15)


def test_drawdown_throttle_accepts_negative_drawdown_convention():
    # rolling_drawdown/max_drawdown NEGATIF deger doner (orn. -0.15);
    # DrawdownThrottle bunu abs() ile normalize etmelidir.
    throttle = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
    assert throttle.multiplier(-0.15) == pytest.approx(0.5)


def test_drawdown_throttle_apply_scales_position_size():
    throttle = DrawdownThrottle([(0.10, 0.5), (0.20, 0.0)])
    assert throttle.apply(1_000.0, 0.15) == pytest.approx(500.0)


def test_drawdown_throttle_rejects_empty_tiers():
    with pytest.raises(ValueError):
        DrawdownThrottle([])


def test_rolling_drawdown_matches_manual_calculation():
    curve = pd.Series([1.0, 1.1, 1.05, 0.9, 1.2, 0.8])
    expected = curve / curve.cummax() - 1.0
    result = rolling_drawdown(curve)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_rolling_drawdown_worst_point_matches_max_drawdown():
    curve = pd.Series([1.0, 1.1, 1.05, 0.9, 1.2, 0.8])
    assert rolling_drawdown(curve).min() == pytest.approx(max_drawdown(curve))


def test_rolling_drawdown_monotonic_increasing_curve_is_zero():
    curve = pd.Series([1.0, 1.05, 1.1, 1.2])
    assert (rolling_drawdown(curve) == 0.0).all()


# --------------------------------------------------------------------------
# drawdown.py - aggregate_portfolio_drawdown
# --------------------------------------------------------------------------


def test_aggregate_portfolio_drawdown_offsetting_strategies_reduces_drawdown():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    a = pd.Series([0.05, -0.05, 0.05, -0.05], index=idx)
    b = pd.Series([-0.05, 0.05, -0.05, 0.05], index=idx)
    combined = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": b}))
    assert combined.min() == pytest.approx(0.0)


def test_aggregate_portfolio_drawdown_correlated_strategies_matches_single():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    a = pd.Series([0.05, -0.05, 0.05, -0.05], index=idx)
    combined = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": a}))
    solo = rolling_drawdown((1 + a).cumprod())
    assert combined.min() == pytest.approx(solo.min())


def test_aggregate_portfolio_drawdown_respects_custom_weights():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    a = pd.Series([0.05, -0.05, 0.05, -0.05], index=idx)
    b = pd.Series([-0.05, 0.05, -0.05, 0.05], index=idx)
    weighted = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": b}), weights={"A": 1.0, "B": 0.0})
    solo = rolling_drawdown((1 + a).cumprod())
    assert weighted.min() == pytest.approx(solo.min())


def test_aggregate_portfolio_drawdown_defaults_to_equal_weights():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    a = pd.Series([0.05, -0.05, 0.05, -0.05], index=idx)
    b = pd.Series([-0.05, 0.05, -0.05, 0.05], index=idx)
    equal_weighted = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": b}))
    explicit = aggregate_portfolio_drawdown(pd.DataFrame({"A": a, "B": b}), weights={"A": 0.5, "B": 0.5})
    pd.testing.assert_series_equal(equal_weighted, explicit, check_names=False)


def test_aggregate_portfolio_drawdown_rejects_empty_columns():
    with pytest.raises(ValueError):
        aggregate_portfolio_drawdown(pd.DataFrame(index=pd.date_range("2026-01-01", periods=3)))


# --------------------------------------------------------------------------
# execution.py - apply_stop_loss_take_profit
# --------------------------------------------------------------------------


def test_apply_sl_tp_finds_stop_loss_exit_before_take_profit():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 102.0, 95.0, 110.0], index=idx)
    result = apply_stop_loss_take_profit(
        prices, entry_price=100.0, entry_index=idx[0], sl_pct=0.05, tp_pct=0.08
    )
    assert result == {"exit_index": idx[2], "exit_price": 95.0, "reason": "stop_loss"}


def test_apply_sl_tp_finds_take_profit_exit():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 103.0, 109.0, 90.0], index=idx)
    result = apply_stop_loss_take_profit(
        prices, entry_price=100.0, entry_index=idx[0], sl_pct=0.05, tp_pct=0.08
    )
    assert result == {"exit_index": idx[2], "exit_price": 109.0, "reason": "take_profit"}


def test_apply_sl_tp_returns_none_reason_when_nothing_triggers():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    result = apply_stop_loss_take_profit(
        prices, entry_price=100.0, entry_index=idx[0], sl_pct=0.05, tp_pct=0.10
    )
    assert result == {"exit_index": idx[-1], "exit_price": 103.0, "reason": "none"}


def test_apply_sl_tp_skips_stop_loss_check_when_sl_pct_is_none():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 80.0, 120.0], index=idx)
    result = apply_stop_loss_take_profit(
        prices, entry_price=100.0, entry_index=idx[0], sl_pct=None, tp_pct=0.10
    )
    # sl_pct=None oldugu icin 80.0'a buyuk dususe ragmen kapanmaz;
    # tp (110) 120.0'da tetiklenir.
    assert result == {"exit_index": idx[2], "exit_price": 120.0, "reason": "take_profit"}


def test_apply_sl_tp_raises_for_entry_index_not_in_series():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0], index=idx)
    with pytest.raises(ValueError):
        apply_stop_loss_take_profit(
            prices, entry_price=100.0, entry_index=pd.Timestamp("2099-01-01"), sl_pct=0.05
        )


# --------------------------------------------------------------------------
# execution.py - TransactionCostModel / square_root_market_impact
# --------------------------------------------------------------------------


def test_transaction_cost_model_matches_formula():
    model = TransactionCostModel(fixed_fee=5.0, pct_fee=0.001, slippage_pct=0.0005)
    trade_value = 100_000.0
    expected = 5.0 + trade_value * (0.001 + 0.0005)
    assert model.cost(trade_value) == pytest.approx(expected)


def test_transaction_cost_model_uses_defaults():
    model = TransactionCostModel()
    assert model.cost(100_000.0) == pytest.approx(100_000.0 * 0.001)


def test_square_root_market_impact_matches_formula():
    trade_value, adv, vol, coeff = 50_000.0, 2_000_000.0, 0.02, 1.2
    expected = coeff * vol * np.sqrt(trade_value / adv)
    assert square_root_market_impact(trade_value, adv, vol, impact_coefficient=coeff) == pytest.approx(
        expected
    )


def test_square_root_market_impact_scales_with_square_root_of_size():
    # trade_value 4 katina cikinca etki 2 katina (sqrt(4)=2) cikmali.
    adv, vol = 1_000_000.0, 0.02
    impact_small = square_root_market_impact(10_000.0, adv, vol)
    impact_large = square_root_market_impact(40_000.0, adv, vol)
    assert impact_large == pytest.approx(impact_small * 2.0)


def test_square_root_market_impact_rejects_non_positive_adv():
    with pytest.raises(ValueError):
        square_root_market_impact(10_000.0, 0.0, 0.02)


# --------------------------------------------------------------------------
# execution.py - realistic_slippage
# --------------------------------------------------------------------------


def test_realistic_slippage_buy_increases_price():
    mult = realistic_slippage(50_000.0, 2_000_000.0, 0.02, side="buy")
    assert mult > 1.0


def test_realistic_slippage_sell_decreases_price():
    mult = realistic_slippage(50_000.0, 2_000_000.0, 0.02, side="sell")
    assert mult < 1.0


def test_realistic_slippage_buy_sell_symmetric_around_one():
    buy = realistic_slippage(50_000.0, 2_000_000.0, 0.02, side="buy")
    sell = realistic_slippage(50_000.0, 2_000_000.0, 0.02, side="sell")
    assert (buy - 1.0) == pytest.approx(1.0 - sell)


def test_realistic_slippage_rejects_invalid_side():
    with pytest.raises(ValueError):
        realistic_slippage(50_000.0, 2_000_000.0, 0.02, side="hold")


# --------------------------------------------------------------------------
# execution.py - simulate_partial_fills
# --------------------------------------------------------------------------


def test_simulate_partial_fills_respects_participation_rate():
    volumes = pd.Series([1000.0, 1000.0, 1000.0])
    prices = pd.Series([10.0, 10.0, 10.0])
    fill = simulate_partial_fills(order_qty=250, bar_volumes=volumes, bar_prices=prices, participation_rate=0.1)
    # her bar'da en fazla 1000*0.1=100 doldurulabilir -> 3 barda 250 tamamen dolar
    assert isinstance(fill, OrderFill)
    assert fill.filled_qty == pytest.approx(250.0)
    assert fill.remaining_qty == pytest.approx(0.0)
    assert fill.fully_filled
    assert fill.bars_to_fill == 3


def test_simulate_partial_fills_leaves_remainder_when_volume_insufficient():
    volumes = pd.Series([100.0, 100.0])
    prices = pd.Series([10.0, 10.0])
    fill = simulate_partial_fills(order_qty=1000, bar_volumes=volumes, bar_prices=prices, participation_rate=0.1)
    # her bar'da en fazla 10 doldurulabilir -> 2 barda sadece 20 dolar, 980 kalir
    assert fill.filled_qty == pytest.approx(20.0)
    assert fill.remaining_qty == pytest.approx(980.0)
    assert not fill.fully_filled


def test_simulate_partial_fills_avg_price_is_volume_weighted():
    volumes = pd.Series([100.0, 100.0])
    prices = pd.Series([10.0, 20.0])
    fill = simulate_partial_fills(order_qty=20, bar_volumes=volumes, bar_prices=prices, participation_rate=0.1)
    # bar1: 10 adet @10, bar2: 10 adet @20 -> agirlikli ortalama 15
    assert fill.avg_price == pytest.approx(15.0)


def test_simulate_partial_fills_rejects_non_positive_qty():
    with pytest.raises(ValueError):
        simulate_partial_fills(0, pd.Series([100.0]), pd.Series([10.0]))


def test_simulate_partial_fills_rejects_invalid_participation_rate():
    with pytest.raises(ValueError):
        simulate_partial_fills(10, pd.Series([100.0]), pd.Series([10.0]), participation_rate=1.5)
