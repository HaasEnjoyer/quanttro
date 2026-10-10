"""
quanttro.live.loop
=======================
Sinyal uretimini broker'a baglayan orkestrasyon katmani.
"""

from __future__ import annotations

from typing import Callable

from .broker import BrokerInterface, Order, OrderSide


class LiveTradingLoop:
    """
    Bir "hedef pozisyon" fonksiyonunu (``signal_fn``) periyodik olarak
    calistirip, mevcut pozisyonlarla hedef arasindaki farki broker
    uzerinden emre donusturen basit bir orkestrasyon katmani.

    Bu sinif NE ZAMAN ("her gun kapanista", "her 5 dakikada") cagrilacagini
    BELIRLEMEZ - zamanlama (cron, websocket bar event'i, vb.) kullanicinin
    sorumlulugundadir; :meth:`step` sadece "bir karar noktasinda ne
    yapilmali" sorusunu cevaplar. Bu ayrim bilerek yapilmistir: zamanlama
    mekanizmasi ortam basina (backtest vs. canli, borsa saatleri, veri
    kaynagi) degisir, karar mantigi degismez.

    Parametreler
    ------------
    broker : BrokerInterface
        Emirlerin iletilecegi broker (``SimulatedBroker`` ya da gercek bir
        implementasyon).
    signal_fn : Callable[..., dict[str, float]]
        Cagrildiginda ``{symbol: hedef_qty}`` sozlugu dondiren fonksiyon
        (orn. bir modelin tahminlerinden pozisyon boyutuna cevrilmis
        hedefler - bkz. :mod:`quanttro.risk.sizing`).
    min_trade_qty : float
        Hedef ile mevcut pozisyon arasindaki farkin, bir emir olusturmak
        icin asmasi gereken esik (gurultu/yuvarlama kaynakli gereksiz
        mikro-emirleri engellemek icin).
    """

    def __init__(
        self,
        broker: BrokerInterface,
        signal_fn: Callable[..., dict[str, float]],
        min_trade_qty: float = 1e-6,
    ):
        self.broker = broker
        self.signal_fn = signal_fn
        self.min_trade_qty = min_trade_qty
        self.last_orders: list[Order] = []

    def step(self, *signal_args, **signal_kwargs) -> list[Order]:
        """
        ``signal_fn``'i cagirir, sonucu mevcut pozisyonlarla karsilastirir,
        gereken alis/satis emirlerini broker'a iletir ve iletilen
        ``Order`` listesini dondurur.
        """
        targets = self.signal_fn(*signal_args, **signal_kwargs)
        current = self.broker.get_positions()

        orders: list[Order] = []
        symbols = set(targets) | set(current)
        for symbol in symbols:
            target_qty = targets.get(symbol, 0.0)
            current_qty = current[symbol].qty if symbol in current else 0.0
            delta = target_qty - current_qty
            if abs(delta) < self.min_trade_qty:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = Order(symbol=symbol, side=side, qty=abs(delta))
            orders.append(self.broker.place_order(order))

        self.last_orders = orders
        return orders
