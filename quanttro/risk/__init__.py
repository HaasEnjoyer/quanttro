"""
quanttro.risk
=================
Pozisyon boyutlandirma ve risk yonetimi araclari.

Alt moduller
------------
sizing
    Pozisyon boyutlandirma yontemleri: fixed fractional, (fractional)
    Kelly, volatilite hedefleme, ATR tabanli risk boyutlandirma ve
    bunlari tek bir arayuzda birlestiren :class:`PositionSizer`.
drawdown
    Equity egrisi (drawdown) bazli pozisyon kisitlama:
    :class:`DrawdownThrottle` ve :func:`rolling_drawdown` yardimci
    fonksiyonu.
execution
    Islem seviyesinde risk araclari: stop-loss/take-profit tetikleme
    tespiti (:func:`apply_stop_loss_take_profit`), islem maliyeti modeli
    (:class:`TransactionCostModel`), basitlestirilmis karekok piyasa etkisi
    modeli (:func:`square_root_market_impact`), yon-farkinda slippage
    (:func:`realistic_slippage`) ve kismi doldurma/order-queuing simulasyonu
    (:func:`simulate_partial_fills`, :class:`OrderFill`).
"""

from __future__ import annotations

from .drawdown import DrawdownThrottle, aggregate_portfolio_drawdown, rolling_drawdown
from .execution import (
    OrderFill,
    TransactionCostModel,
    apply_stop_loss_take_profit,
    realistic_slippage,
    simulate_partial_fills,
    square_root_market_impact,
)
from .sizing import (
    PositionSizer,
    atr_based_size,
    fixed_fractional,
    kelly_fraction,
    volatility_target_size,
)

__all__ = [
    # sizing
    "fixed_fractional",
    "kelly_fraction",
    "volatility_target_size",
    "atr_based_size",
    "PositionSizer",
    # drawdown
    "DrawdownThrottle",
    "rolling_drawdown",
    "aggregate_portfolio_drawdown",
    # execution
    "apply_stop_loss_take_profit",
    "TransactionCostModel",
    "square_root_market_impact",
    "realistic_slippage",
    "simulate_partial_fills",
    "OrderFill",
]
