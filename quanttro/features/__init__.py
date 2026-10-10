"""
quanttro.features
====================
BIST kantitatif arastirma hatti icin ozellik muhendisligi modulu.

Alt moduller
------------
core
    Temel fiyat/hacim ozellikleri: gecikmeli getiriler, hacim z-skoru,
    gerceklesen volatilite, ATR, tavan/taban flag'i, ileri (forward) getiri.
technical
    Klasik teknik analiz indikatorleri: RSI, MACD, Bollinger Bands,
    Stochastic Oscillator, OBV, Momentum.
calendar
    Takvim/zaman bazli ozellikler: haftanin gunu, ay donumu, saat
    (intraday), Pazartesi/Sali kurali.
cross_sectional
    Coklu-hisse (panel) bazli bagil guc / siralama ozellikleri.
"""

from __future__ import annotations

from .calendar import (
    day_of_week_features,
    hour_of_day_features,
    is_monday_or_tuesday,
    turn_of_month_flag,
)
from .core import (
    at_price_limit,
    average_true_range,
    forward_return,
    lagged_returns,
    realized_volatility,
    volume_zscore,
)
from .cross_sectional import (
    cross_sectional_rank,
    relative_forward_target,
    relative_strength,
    sector_relative_strength,
)
from .technical import (
    bollinger_bands,
    macd,
    momentum,
    on_balance_volume,
    rsi,
    stochastic_oscillator,
)

__all__ = [
    # core
    "lagged_returns",
    "volume_zscore",
    "realized_volatility",
    "average_true_range",
    "at_price_limit",
    "forward_return",
    # technical
    "rsi",
    "macd",
    "bollinger_bands",
    "stochastic_oscillator",
    "on_balance_volume",
    "momentum",
    # calendar
    "day_of_week_features",
    "turn_of_month_flag",
    "hour_of_day_features",
    "is_monday_or_tuesday",
    # cross_sectional
    "relative_strength",
    "relative_forward_target",
    "cross_sectional_rank",
    "sector_relative_strength",
]
