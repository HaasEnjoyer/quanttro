"""
quanttro.risk.drawdown
===========================
Equity egrisi (drawdown) bazli pozisyon boyutu kisitlama araclari.

Buradaki fikir, "sermaye su kadar dustuyse pozisyonlari kucult/kapat"
seklindeki kademeli (tiered) risk yonetimini tek bir sinifa
(:class:`DrawdownThrottle`) toplamaktir; :func:`rolling_drawdown` da ona
beslenecek anlik drawdown serisini ureten yardimci fonksiyondur.

:func:`aggregate_portfolio_drawdown` ise BIRDEN FAZLA strateji/parite'nin
getiri serilerini TEK BIR hesap egrisinde birlestirip o birlesik egrinin
drawdown'unu hesaplar - "her biri ayri ayri degil, aynı hesapta birlikte
calistiginda TOPLAM drawdown ne olur" sorusuna cevap verir (tek tek
stratejilerin kendi drawdown'lari kabul edilebilir gorunse de, aynı hesapta
eşzamanlı calistiklarinda TOPLAM drawdown cok daha buyuk olabilir).
"""

from __future__ import annotations

import pandas as pd


class DrawdownThrottle:
    """
    Guncel drawdown seviyesine gore pozisyon buyuklugunu kademeli olarak
    dusuren/durduran basit bir risk yonetim katmani.

    ``tiers``, ``(drawdown_threshold, size_multiplier)`` ikililerinden
    olusan bir listedir; orn. ``[(0.10, 0.5), (0.20, 0.0)]`` = drawdown
    %10'u gectiginde pozisyon yariya (0.5x), %20'yi gectiginde sifira
    (0.0x) iner.

    ``drawdown_threshold`` degerleri BUYUKLUK (pozitif, orn. 0.10 = %10
    dusus) olarak verilir. :meth:`multiplier` ve :meth:`apply`'a verilen
    ``current_drawdown`` de ister pozitif buyukluk ister negatif (orn.
    :func:`rolling_drawdown` ya da
    ``quanttro.backtest.metrics.max_drawdown`` ciktisindaki gibi, -0.15
    = %15 dusus) olsun - icerde ``abs()`` ile normalize edilir.
    """

    def __init__(self, tiers: list[tuple[float, float]]):
        if not tiers:
            raise ValueError("tiers bos olamaz")
        # En kucuk esikten en buyuge dogru sirala - multiplier() hesabinda
        # "gecilen esiklerin EN YUKSEGI"ni bulmayi basitlestirir.
        self.tiers = sorted(tiers, key=lambda tier: tier[0])

    def multiplier(self, current_drawdown: float) -> float:
        """
        Mevcut drawdown'a uygulanacak pozisyon carpanini dondurur.

        ``current_drawdown``'in buyuklugu (``abs()``) esiklerle
        karsilastirilir; esik >= kontrolu ile (``dd >= threshold``) GECILEN
        esiklerin EN YUKSEGININ carpani uygulanir. Hicbir esik gecilmemisse
        (drawdown hepsinden kucukse) 1.0 (tam pozisyon, kisitlama yok)
        doner.
        """
        dd = abs(current_drawdown)
        crossed = [multiplier for threshold, multiplier in self.tiers if dd >= threshold]
        if not crossed:
            return 1.0
        return crossed[-1]

    def apply(self, position_size: float, current_drawdown: float) -> float:
        """``position_size * multiplier(current_drawdown)``."""
        return position_size * self.multiplier(current_drawdown)


def rolling_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """
    Her noktada o ana kadarki kumulatif getirinin running-max'ina gore
    drawdown serisi.

    ``quanttro.backtest.metrics.max_drawdown`` ile AYNI formulu kullanir
    (``drawdown_t = curve_t / running_max_t - 1``), ama tek bir skaler
    (en kotu nokta) degil TAM SERIYI dondurur - :class:`DrawdownThrottle`
    ile birlikte her bar'da guncel drawdown'u beslemek icin kullanilir.

    Donus degerleri NEGATIF ya da sifirdir (orn. -0.15 = %15 dusus);
    :class:`DrawdownThrottle` bunlari dogrudan kabul eder (``abs()`` ile
    normalize eder).
    """
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


def aggregate_portfolio_drawdown(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series | None = None,
) -> pd.Series:
    """
    Birden fazla strateji/parite'nin getiri serilerini (``returns``'un her
    kolonu bir strateji/parite) ``weights`` ile agirliklandirip TEK BIR
    birlesik hesap egrisi kurar, sonra bu birlesik egrinin
    :func:`rolling_drawdown`'unu dondurur.

    Bu, "her stratejiyi kendi basina backtest edip kendi drawdown'una
    bakmak" ile "hepsini AYNI HESAPTA EŞZAMANLI calistirmak" arasindaki
    farki yakalar: stratejilerin kayip gunleri CAKISIRSA (korelasyonlu
    iseler), birlesik hesabin drawdown'u her bir stratejinin TEK BASINA
    drawdown'undan ÇOK DAHA BUYUK olabilir - bu, tek stratejili
    backtest'lerin gormedigi bir risktir.

    Parametreler
    ------------
    returns : pd.DataFrame
        Her kolonu bir strateji/parite'nin periyodik (orn. gunluk)
        yuzdesel getiri serisi oldugu DataFrame. Tum kolonlar AYNI index
        (zaman damgasi) uzerinde hizalanmis olmalidir.
    weights : dict[str, float] | pd.Series | None
        Her kolona verilecek agirlik (orn. sermayenin ne kadari o
        stratejiye ayrilmis). ``None`` ise tum kolonlara esit agirlik
        (``1/n_kolon``) verilir. Toplami 1.0 olmak ZORUNDA DEGILDIR (orn.
        kaldiraçli tahsisler icin >1.0 toplam da gecerlidir).

    Donus
    -----
    pd.Series
        Birlesik (agirlikli toplam) hesabin :func:`rolling_drawdown`
        serisi - NEGATIF ya da sifir degerler (orn. -0.22 = %22 dusus).

    Hatalar
    -------
    ValueError
        ``returns`` bossa (hic kolon yoksa).
    """
    if returns.shape[1] == 0:
        raise ValueError("returns en az bir kolon icermelidir")

    if weights is None:
        weights_series = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    else:
        weights_series = pd.Series(weights).reindex(returns.columns).fillna(0.0)

    combined_returns = (returns * weights_series).sum(axis=1)
    combined_equity = (1.0 + combined_returns).cumprod()
    return rolling_drawdown(combined_equity)
