"""
quanttro.risk.sizing
========================
Pozisyon boyutlandirma (position sizing) yontemleri.

Populer backtest kutuphanelerinde (vectorbt, backtrader) pozisyon
boyutlandirma genellikle dagınık "Sizer" siniflari olarak bulunur; burada
dort klasik yontem (fixed fractional, (fractional) Kelly, volatilite
hedefleme, ATR tabanli risk boyutlandirma) hem bagimsiz fonksiyonlar hem
de tek bir ortak arayuz (:class:`PositionSizer`) olarak sunulur.

Tum fonksiyonlar tek bir skaler (float) girdi/cikti uzerinde calisir -
seri/DataFrame bazli toplu hesaplama gerekiyorsa cagiran taraf
``Series.apply``/vektorel islemlerle sarmalayabilir.
"""

from __future__ import annotations


def fixed_fractional(equity: float, fraction: float = 0.02) -> float:
    """
    Sabit oranli (fixed fractional) pozisyon boyutlandirma.

    Sermayenin (`equity`) sabit bir yuzdesini (`fraction`) pozisyon
    buyuklugu olarak doner: ``position_size = equity * fraction``.

    En basit ve en yaygin boyutlandirma yontemidir; piyasa kosullarina
    (volatilite, kazanma orani vb.) duyarsizdir.
    """
    return equity * fraction


def kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """
    (Fractional) Kelly kriteri ile optimal bahis/pozisyon oranini hesaplar.

    Klasik Kelly formulu::

        f* = win_rate - (1 - win_rate) / win_loss_ratio

    ``f*``, uzun donemde sermaye buyumesini (log-buyume) maksimize eden
    teorik bahis oranidir. Ancak ``f*``:

    - NEGATIF cikabilir (edge yoksa ya da win_loss_ratio cok kotuyse) -
      bu durumda "bahis yapma" anlamina gelir, pozisyon buyuklugu olarak
      kullanilamaz.
    - Gecersiz/asiri girdilerde (orn. ``win_rate`` yanlislikla [0,1]
      disinda verilirse) 1'i de asabilir - sermayenin %100'unden fazlasini
      "pozisyon buyuklugu orani" olarak dondurmek anlamsizdir.

    Bu yuzden ``f*`` once ``[0, 1]`` araligina kirpilir (clip), sonra
    ``kelly_multiplier`` ile olceklenir. ``kelly_multiplier`` varsayilan
    olarak 0.5'tir ("yari-Kelly" / half-Kelly): tam Kelly (f*) teorik
    olarak optimal olsa da pratikte tahmin hatalarina (win_rate ve
    win_loss_ratio asla tam bilinmez) karsi asiri hassastir ve buyuk
    drawdown'lara yol acar; fractional Kelly bu riski bilinen bir bedelle
    (daha dusuk ama daha az oynak buyume orani ile) azaltir.

    Parametreler
    ------------
    win_rate : float
        Kazanma orani (0-1 arasi beklenir).
    win_loss_ratio : float
        Ortalama kazanc / ortalama kayip orani (pozitif beklenir).
    kelly_multiplier : float
        Kirpilmis ``f*``'a uygulanan olcek (varsayilan 0.5 = yari-Kelly).

    Donus
    -----
    float
        ``clip(f*, 0, 1) * kelly_multiplier``.
    """
    f_star = win_rate - (1.0 - win_rate) / win_loss_ratio
    f_star_clipped = min(max(f_star, 0.0), 1.0)
    return f_star_clipped * kelly_multiplier


def volatility_target_size(
    target_vol: float,
    asset_vol: float,
    max_leverage: float = 1.0,
) -> float:
    """
    Volatilite hedefleme (volatility targeting) ile kaldirac/boyut orani.

    Fikir: portfoyun (ya da pozisyonun) gerceklesen oynakliginin sabit bir
    `target_vol` seviyesinde kalmasini saglamak icin, varligin kendi
    oynakligi (`asset_vol`) yukseldiginde pozisyon kuculur, dustugunde
    buyur::

        leverage = target_vol / asset_vol

    Sonuc, asiri kaldiraçlanmayi onlemek icin `max_leverage` ile
    sinirlanir (ustten kirpilir).

    `asset_vol <= 0` (orn. hareketsiz/sabit fiyatli bir seri) ozel bir
    durumdur: matematiksel olarak oran sonsuza gider; bu fonksiyon bu
    durumda (sifira bolme hatasi firlatmak yerine) dogrudan `max_leverage`
    doner - "oynaklik olcumu gecersiz/anlamsiz, en fazla izin verilen
    kaldiraci kullan" seklinde yorumlanabilir.

    Parametreler
    ------------
    target_vol : float
        Hedeflenen (yillliklandirilmis ya da periyot bazli - cagiran
        tarafin tutarli kullanmasi gerekir) oynaklik seviyesi.
    asset_vol : float
        Varligin gerceklesen/tahmini oynakligi (ayni birimde).
    max_leverage : float
        Dondurulecek oranin ustten sinirlandigi deger (varsayilan 1.0 =
        kaldiracsiz).

    Donus
    -----
    float
        ``min(target_vol / asset_vol, max_leverage)``.
    """
    if asset_vol <= 0:
        return max_leverage
    leverage = target_vol / asset_vol
    return float(min(leverage, max_leverage))


def atr_based_size(
    equity: float,
    atr: float,
    risk_per_trade: float = 0.01,
    price: float | None = None,
) -> float:
    """
    ATR (Average True Range) tabanli, risk-paritesi mantigiyla pozisyon
    boyutlandirma.

    Stop-loss mesafesinin yaklasik `atr` kadar olacagini varsayarak,
    pozisyon buyuklugu sermayenin (`equity`) `risk_per_trade` kadarlik
    kismi stop'a çarpıldığında kaybedilecek sekilde hesaplanir::

        position_size = equity * risk_per_trade / atr

    `atr` MUTLAK fiyat biriminde verilirse (orn.
    ``quanttro.features.core.average_true_range(..., normalize=False)``),
    sonuc PAY ADEDI (number of shares) olarak yorumlanir: her pay basina
    stop mesafesi `atr` kadar oldugundan, toplam riski `equity *
    risk_per_trade`'e sabitleyen pay adedi budur.

    `price` verilirse, `atr`'nin NORMALIZE/yuzdesel oldugu varsayilir
    (orn. ``average_true_range(..., normalize=True)`` ciktisi, yani
    atr/close) - bu durumda yukaridaki formul once NOSYONEL (TL) pozisyon
    buyuklugunu verir, ardindan `price`'a bolunerek pay adedine cevrilir::

        shares = (equity * risk_per_trade / atr) / price

    Parametreler
    ------------
    equity : float
        Toplam sermaye.
    atr : float
        Average True Range - mutlak (fiyat birimi, `price=None` ise) ya
        da normalize/yuzdesel (`price` verilirse) olabilir. Pozitif
        olmalidir.
    risk_per_trade : float
        Islem basina riske atilacak sermaye orani (varsayilan %1).
    price : float | None
        Verilirse `atr`'nin normalize oldugu varsayilir ve sonuc pay
        adedine cevrilir. Pozitif olmalidir.

    Donus
    -----
    float
        Pay adedi (ya da `atr` mutlak birimdeyse dogrudan, `price`
        verilirse notional/price seklinde hesaplanan pay adedi).

    Hatalar
    -------
    ValueError
        `atr <= 0` ya da (`price` verilmisse) `price <= 0` durumunda.
    """
    if atr <= 0:
        raise ValueError("atr pozitif olmalidir")
    size = equity * risk_per_trade / atr
    if price is not None:
        if price <= 0:
            raise ValueError("price pozitif olmalidir")
        size = size / price
    return float(size)


class PositionSizer:
    """
    Yukaridaki dort pozisyon boyutlandirma yontemini tek bir ortak
    arayuz altinda birlestiren sinif.

    ``__init__``'te secilen ``method`` ("fixed_fractional" | "kelly" |
    "volatility_target" | "atr_based") icin gereken parametreler ister
    ``__init__(**params)``'ta (bir kez, tum cagrilar icin ortak varsayilan
    olarak), ister her ``.size(equity, **kwargs)`` cagrisinda (o cagriya
    ozel, ``params``'in uzerine yazarak) verilebilir.

    Her yontemin GEREKTIRDIGI (doğal bir varsayimi olmayan) parametreler
    verilmezse (ne ``params``'ta ne ``kwargs``'ta), sessizce bir varsayim
    yapilmaz - acik bir ``ValueError`` firlatilir. (Orn. "kelly" yontemi
    icin ``win_rate``/``win_loss_ratio`` zorunludur; ``kelly_multiplier``
    ise 0.5 varsayilanina dusebilir.)

    ``.size()`` her zaman "equity ile olceklenmis" nihai pozisyon
    buyuklugunu (ayni birimde, orn. TL) doner - :func:`kelly_fraction` ve
    :func:`volatility_target_size` tek basina bir ORAN/CARPAN dondurdugu
    icin, ``PositionSizer`` bunlari ``equity`` ile çarparak diger iki
    yontemle (:func:`fixed_fractional`, :func:`atr_based_size`) ayni
    "pozisyon buyuklugu" semantigine getirir.
    """

    _VALID_METHODS = ("fixed_fractional", "kelly", "volatility_target", "atr_based")

    def __init__(self, method: str, **params):
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Gecersiz method: {method!r}. Gecerli degerler: {self._VALID_METHODS}"
            )
        self.method = method
        self.params = params

    def _require(self, merged: dict, keys: tuple[str, ...]) -> None:
        missing = [k for k in keys if k not in merged]
        if missing:
            raise ValueError(
                f"'{self.method}' yontemi icin eksik parametre(ler): {missing}. "
                f"PositionSizer(method={self.method!r}, ...) olusturulurken ya da "
                f".size(equity, ...) cagrisinda verilmelidir."
            )

    def size(self, equity: float, **kwargs) -> float:
        """
        Secilen yonteme (``self.method``) gore pozisyon buyuklugunu
        hesaplar.

        ``self.params`` (``__init__``'te verilen) ile ``kwargs`` (bu
        cagriya ozel verilen) birlesir; ayni isimli bir parametre varsa
        ``kwargs`` kazanir.
        """
        merged = {**self.params, **kwargs}

        if self.method == "fixed_fractional":
            fraction = merged.get("fraction", 0.02)
            return fixed_fractional(equity, fraction=fraction)

        if self.method == "kelly":
            self._require(merged, ("win_rate", "win_loss_ratio"))
            kelly_multiplier = merged.get("kelly_multiplier", 0.5)
            fraction = kelly_fraction(
                merged["win_rate"],
                merged["win_loss_ratio"],
                kelly_multiplier=kelly_multiplier,
            )
            return equity * fraction

        if self.method == "volatility_target":
            self._require(merged, ("target_vol", "asset_vol"))
            max_leverage = merged.get("max_leverage", 1.0)
            leverage = volatility_target_size(
                merged["target_vol"],
                merged["asset_vol"],
                max_leverage=max_leverage,
            )
            return equity * leverage

        if self.method == "atr_based":
            self._require(merged, ("atr",))
            risk_per_trade = merged.get("risk_per_trade", 0.01)
            price = merged.get("price")
            return atr_based_size(
                equity,
                merged["atr"],
                risk_per_trade=risk_per_trade,
                price=price,
            )

        raise ValueError(f"Gecersiz method: {self.method!r}")  # pragma: no cover
