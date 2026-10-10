"""
quanttro.live.broker
=========================
Emir/pozisyon temsili ve broker arayuzu.

``BrokerInterface`` soyut sinifinin tasarim amaci: bir stratejinin sinyal
uretim kodu, kagit islemden gercek bir araci kuruma geçerken DEGISMESIN.
Gercek bir entegrasyon icin yapilmasi gereken, ``BrokerInterface``'i o
kurumun API'sine gore implemente etmektir (bu kutuphane bunu saglamaz -
bkz. :mod:`quanttro.live` modul dokstring'i).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from quanttro.risk.execution import TransactionCostModel, realistic_slippage

_order_id_counter = itertools.count(1)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Bir emrin temsili."""

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    order_id: int = field(default_factory=lambda: next(_order_id_counter))
    status: str = "pending"  # pending | filled | partially_filled | cancelled
    filled_qty: float = 0.0
    avg_fill_price: float | None = None


@dataclass
class Position:
    """Acik bir pozisyonun temsili."""

    symbol: str
    qty: float
    avg_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.avg_price


class BrokerInterface(ABC):
    """
    Gercek/kagit tum broker implementasyonlarinin uymasi gereken soyut
    arayuz. Bir strateji bu arayuze yazilir; alt-sinif degisse de
    (``SimulatedBroker`` -> gercek bir araci kurum implementasyonu)
    strateji kodu degismez.
    """

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Emri iletir ve guncellenmis (``status``/``filled_qty`` dolu)
        ``Order`` nesnesini dondurur."""

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """Bekleyen bir emri iptal eder. Basariliysa ``True`` doner."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Sembol -> ``Position`` sozlugunu dondurur."""

    @abstractmethod
    def get_cash(self) -> float:
        """Kullanilabilir nakit bakiyeyi dondurur."""

    def get_account_value(self, mark_prices: dict[str, float] | None = None) -> float:
        """Nakit + acik pozisyonlarin (mark_prices varsa onunla, yoksa
        ``avg_price`` ile) toplam degeri."""
        positions = self.get_positions()
        mark_prices = mark_prices or {}
        positions_value = sum(
            pos.qty * mark_prices.get(sym, pos.avg_price) for sym, pos in positions.items()
        )
        return self.get_cash() + positions_value


class SimulatedBroker(BrokerInterface):
    """
    ``BrokerInterface``'in kagit-islem (paper trading) implementasyonu.

    Her ``place_order`` cagrisinda, verilen ``price_feed`` fonksiyonundan
    (``symbol -> fiyat``) o anki fiyati alir, isteğe bagli olarak
    :func:`quanttro.risk.execution.realistic_slippage` ile yon-farkinda
    slippage uygular, :class:`quanttro.risk.execution.TransactionCostModel`
    ile komisyon/slippage maliyetini dusurur ve pozisyon/nakit durumunu
    guncelleyip emri TAM doldurulmus olarak isaretler (kismi fill/order
    queuing simulasyonu icin dogrudan
    :func:`quanttro.risk.execution.simulate_partial_fills` kullanin -
    bu, coklu-bar'lik bir hesaplamadir ve tek-bar'lik bu broker
    arayuzunun disindadir).

    Parametreler
    ------------
    price_feed : Callable[[str], float]
        Bir sembol icin "o anki" fiyati dondiren fonksiyon (orn. en son
        kapanis fiyati, ya da canli bir websocket feed'den beslenen bir
        closure).
    initial_cash : float
        Baslangic nakit bakiyesi.
    cost_model : TransactionCostModel | None
        Komisyon/slippage modeli (``None`` ise maliyetsiz kabul edilir -
        SADECE hizli prototipleme icin, gercekci backtest icin bir model
        verilmesi onerilir).
    market_impact_kwargs : dict | None
        Verilirse (``avg_daily_volume_value``, ``volatility`` iceren bir
        sozluk), her emirde ek olarak :func:`realistic_slippage` uygulanir.
    """

    def __init__(
        self,
        price_feed,
        initial_cash: float = 100_000.0,
        cost_model: TransactionCostModel | None = None,
        market_impact_kwargs: dict | None = None,
    ):
        self.price_feed = price_feed
        self.cash = float(initial_cash)
        self.cost_model = cost_model
        self.market_impact_kwargs = market_impact_kwargs
        self._positions: dict[str, Position] = {}
        self._order_history: list[Order] = []

    def place_order(self, order: Order) -> Order:
        price = float(self.price_feed(order.symbol))
        trade_value = price * order.qty

        if self.market_impact_kwargs is not None:
            multiplier = realistic_slippage(
                trade_value=trade_value,
                side=order.side.value,
                **self.market_impact_kwargs,
            )
            price *= multiplier

        cost = self.cost_model.cost(trade_value) if self.cost_model is not None else 0.0

        signed_qty = order.qty if order.side == OrderSide.BUY else -order.qty
        cash_delta = -signed_qty * price - cost
        if order.side == OrderSide.BUY and -cash_delta > self.cash:
            order.status = "cancelled"
            self._order_history.append(order)
            return order

        self.cash += cash_delta
        existing = self._positions.get(order.symbol)
        if existing is None:
            self._positions[order.symbol] = Position(order.symbol, signed_qty, price)
        else:
            new_qty = existing.qty + signed_qty
            if new_qty == 0:
                del self._positions[order.symbol]
            else:
                # Pozisyon yon degistirmiyorsa agirlikli ortalama maliyet;
                # yon degistiriyorsa (orn. long'dan short'a) yeni fiyat baz alinir.
                same_direction = (existing.qty > 0) == (signed_qty > 0)
                if same_direction:
                    total_cost = existing.qty * existing.avg_price + signed_qty * price
                    avg_price = total_cost / new_qty
                else:
                    avg_price = price
                self._positions[order.symbol] = Position(order.symbol, new_qty, avg_price)

        order.status = "filled"
        order.filled_qty = order.qty
        order.avg_fill_price = price
        self._order_history.append(order)
        return order

    def cancel_order(self, order_id: int) -> bool:
        # SimulatedBroker emirleri aninda dolduruyor; bekleyen emir yok.
        return False

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_cash(self) -> float:
        return self.cash

    @property
    def order_history(self) -> list[Order]:
        return list(self._order_history)
