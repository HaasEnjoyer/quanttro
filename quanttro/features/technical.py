"""
quanttro.features.technical
===============================
Klasik teknik analiz indikatorleri.

Bu indikatorler `feature_engineering.py` icinde YOKTU - bu modul mevcut
ozellik setini standart teknik analiz indikatorleriyle zenginlestirir.
Tum formuller klasik/standart tanimlarina gore uygulanmistir:

    - RSI       : Wilder'in Relative Strength Index'i (ussel yumusatma)
    - MACD      : fast/slow EMA farki + signal EMA'si
    - Bollinger : N-periyotluk SMA +- n_std * rolling std
    - Stochastic: %K (kapanisin N-periyotluk high/low araligindaki konumu) + %D (%K'nin SMA'si)
    - OBV       : yon-isaretli hacmin kumulatif toplami
    - Momentum  : N bar onceki fiyata gore fark

Tum fonksiyonlar girdiyle ayni index'i korur; isinma (warm-up) donemindeki
NaN degerler kasten birakilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI), Wilder'in ussel yumusatma yontemiyle.

    1. delta = close.diff()
    2. gain = max(delta, 0), loss = max(-delta, 0)
    3. avg_gain / avg_loss: alpha = 1/window olan ussel hareketli ortalama
       (Wilder'in recursive yumusatma formuluyle matematiksel olarak
       esdeger: avg_t = avg_(t-1) + (x_t - avg_(t-1)) / window)
    4. RS = avg_gain / avg_loss
    5. RSI = 100 - 100 / (1 + RS)

    Donus degeri [0, 100] araligindadir. Kayip hic gorulmediyse (avg_loss
    == 0, orn. surekli yukselen fiyat serisi) RSI = 100 kabul edilir;
    bu, RS = sonsuz durumuna karsilik gelir.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    result = result.where(avg_loss != 0, 100.0)
    result.name = f"rsi_{window}"
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    macd      = EMA(close, fast) - EMA(close, slow)
    signal    = EMA(macd, signal)
    histogram = macd - signal

    EMA'lar `adjust=False` ile hesaplanir (standart, "online" ussel
    hareketli ortalama tanimi: ema_t = alpha*x_t + (1-alpha)*ema_(t-1)).

    Donus: "macd", "signal", "histogram" kolonlarini iceren pd.DataFrame.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: basit hareketli ortalama (SMA) etrafinda +-n_std
    standart sapma genisliginde bantlar.

    middle = rolling_mean(close, window)
    upper  = middle + n_std * rolling_std(close, window)
    lower  = middle - n_std * rolling_std(close, window)
    %b     = (close - lower) / (upper - lower)   (fiyatin bant icindeki
             konumu: 0 = alt banda yapisik, 1 = ust banda yapisik,
             0.5 = ortalamanin tam uzerinde)

    Donus: "upper", "middle", "lower", "%b" kolonlarini iceren pd.DataFrame.
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "%b": percent_b,
    })


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator: kapanisin son `window` barlik high/low
    araligindaki yuzdesel konumu.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = %K'nin `d_window` barlik basit hareketli ortalamasi (sinyal cizgisi)

    Donus: "%K", "%D" kolonlarini iceren pd.DataFrame.
    """
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    percent_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(d_window).mean()
    return pd.DataFrame({
        "%K": percent_k,
        "%D": percent_d,
    })


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV): yon-isaretli hacmin kumulatif toplami.

    OBV_t = OBV_(t-1) + volume_t   eger close_t > close_(t-1)
    OBV_t = OBV_(t-1) - volume_t   eger close_t < close_(t-1)
    OBV_t = OBV_(t-1)              eger close_t == close_(t-1)

    Ilk bar icin onceki kapanis olmadigindan yon 0 kabul edilir (OBV 0'dan
    baslar). OBV'nin mutlak seviyesi degil TRENDI/sekli anlamli oldugundan
    bu baslangic kabulu standart kullanimla tam uyumludur.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv.name = "obv"
    return obv


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """
    Fiyat momentumu: `window` bar onceki kapanisa gore FARK (yuzde degil).

    momentum_t = close_t - close_(t-window)
    """
    mom = close.diff(window)
    mom.name = f"momentum_{window}"
    return mom
