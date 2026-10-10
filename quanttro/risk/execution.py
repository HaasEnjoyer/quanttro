"""
quanttro.risk.execution
============================
Islem (execution) seviyesinde risk araclari.

Uc parca:

- :func:`apply_stop_loss_take_profit`: acik bir pozisyonun fiyat
  serisinde stop-loss/take-profit seviyelerinin NE ZAMAN tetiklendigini
  bulan tarama fonksiyonu.
- :class:`TransactionCostModel`: sabit + yuzdesel komisyon + slippage
  seklinde basit bir islem maliyeti modeli.
- :func:`square_root_market_impact`: basitlestirilmis Almgren-Chriss
  karekok piyasa etkisi modeli.
- :func:`simulate_partial_fills`: bir emrin birden fazla bar'a yayilarak
  KISMI olarak nasil doldugunu simule eder (katilim orani / participation
  rate kisiti ile) - "vectorized backtester'lar kismi fill/order queuing
  modellemiyor" eksikligini kapatir.
- :func:`realistic_slippage`: yon-farkinda (buy/sell) slippage - piyasa
  etkisini ``square_root_market_impact`` ile hesaplayip isaretini emrin
  yonune gore uygular (alis fiyati yukari kayar, satis fiyati asagi kayar).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def apply_stop_loss_take_profit(
    prices: pd.Series,
    entry_price: float,
    entry_index,
    sl_pct: float | None = None,
    tp_pct: float | None = None,
) -> dict:
    """
    ``entry_index``'ten itibaren ``prices`` serisini tarayip, stop-loss
    (``sl_pct``) ya da take-profit (``tp_pct``) seviyesine ilk ulasan
    noktayi bulur.

    ``sl_pct``/``tp_pct`` ``None`` ise o kontrol ATLANIR (orn. sadece
    ``tp_pct`` verilirse stop-loss hic kontrol edilmez, fiyat ne kadar
    duserse dussun). Ikisi de ``None`` ise islem hicbir zaman kapanmaz
    (``reason="none"``).

    Tarama ``entry_index``'ten SONRAKI barlardan baslar (giris barinin
    kendisi kontrol edilmez). Bir barda HEM stop-loss HEM take-profit
    seviyesi gecilmis olsa bile (orn. yuksek oynaklik nedeniyle), gercek
    bar-ici (intrabar) sirayi bilmedigimiz icin KONSERVATIF varsayimla
    stop-loss ONCELIKLI kabul edilir.

    Parametreler
    ------------
    prices : pd.Series
        ``entry_index``'i de iceren fiyat serisi.
    entry_price : float
        Pozisyona giris fiyati.
    entry_index :
        ``prices.index`` icinde giris barinin etiketi (label).
    sl_pct : float | None
        Stop-loss yuzdesi (orn. 0.02 = giristen %2 asagisi, yani
        ``entry_price * (1 - sl_pct)``). ``None`` ise kontrol edilmez.
    tp_pct : float | None
        Take-profit yuzdesi (orn. 0.03 = giristen %3 yukarisi, yani
        ``entry_price * (1 + tp_pct)``). ``None`` ise kontrol edilmez.

    Donus
    -----
    dict
        ``{"exit_index": ..., "exit_price": ..., "reason": "stop_loss" |
        "take_profit" | "none"}``. Hicbir esik tetiklenmezse
        ``exit_index``/``exit_price`` serinin SON elemanidir (henuz acik
        pozisyonun "mark-to-market" cikisi gibi yorumlanabilir),
        ``reason="none"``.

    Hatalar
    -------
    ValueError
        ``entry_index``, ``prices.index`` icinde bulunamazsa.
    """
    if entry_index not in prices.index:
        raise ValueError("entry_index, prices serisinin index'inde bulunamadi")

    position = prices.index.get_loc(entry_index)
    after_entry = prices.iloc[position + 1 :]

    sl_price = entry_price * (1 - sl_pct) if sl_pct is not None else None
    tp_price = entry_price * (1 + tp_pct) if tp_pct is not None else None

    for idx, price in after_entry.items():
        hit_sl = sl_price is not None and price <= sl_price
        hit_tp = tp_price is not None and price >= tp_price
        if hit_sl:
            return {"exit_index": idx, "exit_price": price, "reason": "stop_loss"}
        if hit_tp:
            return {"exit_index": idx, "exit_price": price, "reason": "take_profit"}

    if len(after_entry) == 0:
        return {"exit_index": entry_index, "exit_price": entry_price, "reason": "none"}
    return {
        "exit_index": after_entry.index[-1],
        "exit_price": after_entry.iloc[-1],
        "reason": "none",
    }


class TransactionCostModel:
    """
    Basit islem maliyeti modeli: sabit komisyon + yuzdesel komisyon +
    slippage, hepsi islem degeri (``trade_value``) uzerinden.
    """

    def __init__(
        self,
        fixed_fee: float = 0.0,
        pct_fee: float = 0.001,
        slippage_pct: float = 0.0,
    ):
        self.fixed_fee = fixed_fee
        self.pct_fee = pct_fee
        self.slippage_pct = slippage_pct

    def cost(self, trade_value: float) -> float:
        """
        Toplam islem maliyeti::

            cost = fixed_fee + trade_value * (pct_fee + slippage_pct)
        """
        return self.fixed_fee + trade_value * (self.pct_fee + self.slippage_pct)


def square_root_market_impact(
    trade_value: float,
    avg_daily_volume_value: float,
    volatility: float,
    impact_coefficient: float = 1.0,
) -> float:
    """
    Basitlestirilmis (simplified) Almgren-Chriss karekok piyasa etkisi
    modeli.

    DIKKAT: Bu, akademik literaturdeki tam Almgren-Chriss modelinin
    (gecici/kalici etki ayrimi, islem hizi, risk-aversion parametresi gibi
    bircok ek terim iceren) COK BASITLESTIRILMIS, VARSAYIMSAL bir
    yaklasimidir. Gercek piyasa etkisi varlik/piyasa/zaman dilimine gore
    buyuk farklilik gosterir; buradaki sabit ``impact_coefficient``
    gercek bir kalibrasyonu TEMSIL ETMEZ, sadece oransal bir buyutme
    katsayisidir.

    Formul::

        impact = impact_coefficient * volatility * sqrt(trade_value / avg_daily_volume_value)

    Yani piyasa etkisi, islem buyuklugunun gunluk ortalama islem
    hacmine (``avg_daily_volume_value``, TL cinsinden) oranının
    KAREKOKU ile (dogrusal degil, sqrt ile) ve varligin oynakligi
    (``volatility``) ile olceklenir.

    Parametreler
    ------------
    trade_value : float
        Islemin TL (ya da tutarli herhangi bir para birimi) cinsinden
        degeri.
    avg_daily_volume_value : float
        Varligin ortalama gunluk islem hacmi degeri (ayni birimde).
        Pozitif olmalidir.
    volatility : float
        Varligin oynakligi (periyot bazli, cagiran tarafin tutarli
        kullanmasi gerekir).
    impact_coefficient : float
        Modelin olcek katsayisi (varsayilan 1.0).

    Donus
    -----
    float
        Tahmini piyasa etkisi (fiyat uzerindeki oransal/yuzdesel kayma,
        orn. 0.002 = %0.2).

    Hatalar
    -------
    ValueError
        ``avg_daily_volume_value <= 0`` durumunda.
    """
    if avg_daily_volume_value <= 0:
        raise ValueError("avg_daily_volume_value pozitif olmalidir")
    return float(
        impact_coefficient * volatility * np.sqrt(trade_value / avg_daily_volume_value)
    )


def realistic_slippage(
    trade_value: float,
    avg_daily_volume_value: float,
    volatility: float,
    side: str,
    impact_coefficient: float = 1.0,
) -> float:
    """
    :func:`square_root_market_impact`'i YON-FARKINDA (direction-aware) hale
    getirir: alis (``side="buy"``) emri piyasa etkisi kadar YUKARI, satis
    (``side="sell"``) emri AŞAĞI kayar. Doner deger, emrin gerceklesecegi
    fiyata uygulanacak carpandir (orn. 1.002 -> fiyat %0.2 yukari kaydi).

    Parametreler
    ------------
    trade_value, avg_daily_volume_value, volatility, impact_coefficient :
        bkz. :func:`square_root_market_impact`.
    side : str
        ``"buy"`` ya da ``"sell"`` (buyuk/kucuk harf duyarsiz).

    Donus
    -----
    float
        Fiyata uygulanacak carpan (1.0 etrafinda).

    Hatalar
    -------
    ValueError
        ``side`` "buy"/"sell" disinda bir deger ise.
    """
    side_normalized = side.lower()
    if side_normalized not in ("buy", "sell"):
        raise ValueError('side "buy" ya da "sell" olmalidir')
    impact = square_root_market_impact(
        trade_value, avg_daily_volume_value, volatility, impact_coefficient
    )
    sign = 1.0 if side_normalized == "buy" else -1.0
    return float(1.0 + sign * impact)


@dataclass
class OrderFill:
    """:func:`simulate_partial_fills` sonucu: bir emrin doldurulma ozeti."""

    filled_qty: float
    avg_price: float
    remaining_qty: float
    bars_to_fill: int
    fill_prices: list[float] = field(default_factory=list)
    fill_qtys: list[float] = field(default_factory=list)

    @property
    def fully_filled(self) -> bool:
        return self.remaining_qty <= 0


def simulate_partial_fills(
    order_qty: float,
    bar_volumes: pd.Series,
    bar_prices: pd.Series,
    participation_rate: float = 0.1,
    max_bars: int | None = None,
) -> OrderFill:
    """
    Buyuk bir emrin, vectorized backtester'larin genelde atladigi KISMI
    doldurma (partial fill) / order-queuing davranisini simule eder: her
    bar'da emrin en fazla ``bar_volume * participation_rate`` kadarlik
    kismi doldurulur, kalan kisim sonraki bar'a tasir.

    Bu, "bir emir tek bar'da aninda ve tam olarak gerceklesir" varsayimini
    gevsetir - buyuk emirler (gunluk hacmin onemli bir orani) icin bu
    varsayim gercekci degildir ve gercek piyasa etkisini hafife alir.

    Parametreler
    ------------
    order_qty : float
        Doldurulacak toplam emir buyuklugu (adet, pozitif).
    bar_volumes : pd.Series
        Emrin baslangicindan itibaren sirali bar hacimleri.
    bar_prices : pd.Series
        ``bar_volumes`` ile ayni index'e sahip, o bar'da gerceklesecek
        (orn. kapanis ya da VWAP) fiyat serisi.
    participation_rate : float
        Bir bar'da doldurulabilecek azami hacim orani (varsayilan 0.1 =
        %10 - yaygin bir "likiditeyi bozmama" kurali). ``(0, 1]`` araliginda
        olmalidir.
    max_bars : int | None
        Taranacak azami bar sayisi. ``None`` ise ``bar_volumes``'un tamami
        taranir.

    Donus
    -----
    OrderFill
        Doldurulan miktar, agirlikli ortalama fiyat, kalan miktar ve
        kac bar surdugu.

    Hatalar
    -------
    ValueError
        ``order_qty <= 0`` ya da ``participation_rate`` ``(0, 1]`` disinda ise.
    """
    if order_qty <= 0:
        raise ValueError("order_qty pozitif olmalidir")
    if not (0 < participation_rate <= 1):
        raise ValueError("participation_rate (0, 1] araliginda olmalidir")

    remaining = float(order_qty)
    fill_prices: list[float] = []
    fill_qtys: list[float] = []
    bars_used = 0

    series = bar_volumes if max_bars is None else bar_volumes.iloc[:max_bars]
    for idx, volume in series.items():
        if remaining <= 0:
            break
        bars_used += 1
        bar_capacity = float(volume) * participation_rate
        fill_qty = min(remaining, bar_capacity)
        if fill_qty <= 0:
            continue
        fill_prices.append(float(bar_prices.loc[idx]))
        fill_qtys.append(fill_qty)
        remaining -= fill_qty

    filled = float(order_qty) - remaining
    avg_price = (
        float(np.average(fill_prices, weights=fill_qtys)) if fill_qtys else float("nan")
    )
    return OrderFill(
        filled_qty=filled,
        avg_price=avg_price,
        remaining_qty=remaining,
        bars_to_fill=bars_used,
        fill_prices=fill_prices,
        fill_qtys=fill_qtys,
    )
