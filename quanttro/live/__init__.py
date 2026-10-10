"""
quanttro.live
==================
Canli/kagit (paper) islem icin KOPRU (bridge) katmani.

ONEMLI - KAPSAM: Bu modul hicbir gercek BIST araci kurum API'sine
BAGLI DEGILDIR (boyle bir entegrasyon, her araci kurumun kendi REST/FIX
API'sine, API anahtarina ve regulasyon/uyum surecine baglidir - bu kutuphane
kapsaminin disindadir). Bunun yerine saglanan sey: strateji sinyali ile
"emir ver / pozisyon al" arasindaki arayuzu (:class:`BrokerInterface`)
standartlastirmak, boylece (a) :class:`SimulatedBroker` ile GERCEKCI
maliyet/slippage/kismi-fill modeliyle kagit islem yapilabilir, (b) gercek
bir araci kuruma baglanmak istendiginde tek yapilmasi gereken
``BrokerInterface``'i o kurumun API'sine gore implemente etmek olur -
stratejinin kendisinin (sinyal uretimi) DEGISMEMESI saglanir.

Alt parcalar
------------
- :class:`Order`, :class:`OrderSide`, :class:`OrderType`: emir temsili.
- :class:`Position`: acik pozisyon temsili.
- :class:`BrokerInterface`: soyut (abstract) arayuz - gercek bir araci
  kuruma baglanmak icin bu sinifin alt-sinifi yazilir.
- :class:`SimulatedBroker`: ``BrokerInterface``'in kagit-islem (paper
  trading) implementasyonu - :mod:`quanttro.risk.execution`'daki
  ``TransactionCostModel``/``realistic_slippage``/``simulate_partial_fills``
  araclarini kullanarak GERCEKCI fill fiyati uretir.
- :class:`LiveTradingLoop`: bir sinyal fonksiyonunu ("hedef pozisyonlar")
  periyodik olarak calistirip, mevcut pozisyonlarla farkini alip broker
  uzerinden emir veren orkestrasyon katmani.
"""

from __future__ import annotations

from .broker import (
    BrokerInterface,
    Order,
    OrderSide,
    OrderType,
    Position,
    SimulatedBroker,
)
from .loop import LiveTradingLoop

__all__ = [
    "Order",
    "OrderSide",
    "OrderType",
    "Position",
    "BrokerInterface",
    "SimulatedBroker",
    "LiveTradingLoop",
]
