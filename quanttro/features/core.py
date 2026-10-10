"""
quanttro.features.core
=========================
BIST hisse senedi/endeks serileri icin temel fiyat ve hacim ozellikleri.

Bu modul, proje kokundeki `feature_engineering.py` dosyasindaki orijinal
mantigin genellenmis (parametrik, tekrar kullanilabilir) halidir. Iki onemli
tasarim karari oradan aynen tasinmistir:

    - RETURN_CLIP ile winsorize: piyasa-geneli sok barlarinin (orn. 19 Mart
      2025 gibi ani devaluasyon/satis dalgalari) getiri dagilimini bozmasini
      onlemek icin getiriler [-clip, +clip] araligina kirpilir. Bu, modelin
      tek bir ekstrem gunden asiri etkilenmesini onler.
    - BIST'te gunluk fiyat hareketi tavan/taban kurallari nedeniyle pratikte
      +-%10 ile sinirlidir; LIMIT_THRESHOLD (0.095) bu sinira yakin barlari
      guvenlik payiyla "tavan/taban" olarak isaretler.

Tum fonksiyonlar pandas Series/DataFrame alir ve girdiyle AYNI index'i
korur; NaN degerler (orn. rolling pencere henuz isinmadan once) kasten
birakilir - hangi satirlarin dusurulecegine (dropna) karar vermek cagiran
tarafin sorumlulugundadir (gelecek sizintisini onlemek icin bu bilincli
tercih edilmistir).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 5, 10, 20)
DEFAULT_RETURN_CLIP = 0.12       # feature_engineering.py: RETURN_CLIP
DEFAULT_LIMIT_THRESHOLD = 0.095  # feature_engineering.py: LIMIT_THRESHOLD


def lagged_returns(
    close: pd.Series,
    lags: Sequence[int] = DEFAULT_LAGS,
    clip: float | None = DEFAULT_RETURN_CLIP,
) -> pd.DataFrame:
    """
    Verilen gecikmeler (lag/bar sayisi) icin yuzde getiri kolonlari uretir.

    Her `n` icin `close.pct_change(n)` hesaplanir; `clip` None degilse
    sonuc [-clip, +clip] araliginda winsorize edilir (feature_engineering.py
    RETURN_CLIP mantigi: piyasa-geneli sok barlarini yumusatmak icin).

    Parametreler
    ------------
    close : pd.Series
        Kapanis fiyati serisi.
    lags : Sequence[int]
        Hesaplanacak gecikme (bar) sayilari. Her biri icin "ret_{n}" kolonu
        uretilir.
    clip : float | None
        Winsorize esigi. None verilirse kirpma yapilmaz (ham getiri).

    Donus
    -----
    pd.DataFrame
        Kolonlari "ret_{n}" formatinda, index'i `close` ile ayni.
    """
    out = pd.DataFrame(index=close.index)
    for n in lags:
        ret = close.pct_change(n)
        if clip is not None:
            ret = ret.clip(-clip, clip)
        out[f"ret_{n}"] = ret
    return out


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Hacmin kendi son `window` barlik ortalama/standart sapmasina gore
    z-skoru: (volume - rolling_mean) / rolling_std.

    Anormal yuksek/dusuk hacim barlarini tespit etmek icin kullanilir
    (ornegin calendar.is_monday_or_tuesday ile birlikte Pazartesi/Sali +
    yuksek hacim kuralinin girdisi olarak).
    """
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    z = (volume - mean) / std
    z.name = "volume_zscore"
    return z


def realized_volatility(returns: pd.Series, window: int = 10) -> pd.Series:
    """
    Getirilerin son `window` barlik gerceklesen (realized) volatilitesi
    (ornek standart sapma, ddof=1).
    """
    vol = returns.rolling(window).std()
    vol.name = f"realized_vol_{window}"
    return vol


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    normalize: bool = True,
) -> pd.Series:
    """
    Average True Range (ATR): Wilder'in volatilite olcusu.

    True Range (TR) = max(
        high - low,
        |high - onceki_kapanis|,
        |low - onceki_kapanis|,
    )
    ATR = TR'nin son `window` barlik basit hareketli ortalamasi
    (feature_engineering.py'daki mevcut implementasyonla uyumlu olarak
    Wilder'in ussel/smoothed ortalamasi degil, basit rolling mean
    kullanilir).

    `normalize=True` (varsayilan) ise ATR kapanis fiyatina bolunerek
    fiyat seviyesinden bagimsiz (yuzdesel) hale getirilir - boylece farkli
    fiyat seviyelerindeki hisseler (ornegin 5 TL ile 500 TL) birbiriyle
    karsilastirilabilir olur (feature_engineering.py'daki atr_14 ile ayni
    mantik).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean()
    if normalize:
        atr = atr / close
    atr.name = f"atr_{window}"
    return atr


def at_price_limit(returns: pd.Series, threshold: float = DEFAULT_LIMIT_THRESHOLD) -> pd.Series:
    """
    BIST tavan/taban flag'i: |getiri| >= threshold olan barlari 1 isaretler.

    BIST gunluk fiyat hareketi tavan/taban kurallariyla pratikte +-%10 ile
    sinirlidir; 0.095 esigi (feature_engineering.py'daki LIMIT_THRESHOLD)
    yuvarlama/mikro-fark paylarini goz onunde bulundurarak guvenlik payi
    birakir.
    """
    flag = (returns.abs() >= threshold).astype(int)
    flag.name = "at_limit"
    return flag


def forward_return(close: pd.Series, horizon: int = 1, clip: float | None = None) -> pd.Series:
    """
    `horizon` bar ileriye bakan getiri (hedef degisken / label).

    fwd_ret_t = close_(t+horizon) / close_t - 1

    DIKKAT: Bu fonksiyon kasten GELECEGE bakar (shift(-horizon)) - sadece
    egitim hedefi (target) uretmek icin kullanilmalidir, ASLA bir "feature"
    olarak modele girdi yapilmamalidir (gelecek sizintisi / look-ahead bias).

    `clip` None degilse sonuc [-clip, +clip] araliginda winsorize edilir.
    """
    fwd = close.shift(-horizon) / close - 1
    if clip is not None:
        fwd = fwd.clip(-clip, clip)
    fwd.name = f"fwd_ret_{horizon}"
    return fwd
