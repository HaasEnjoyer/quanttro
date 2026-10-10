"""
quanttro.portfolio.compare
===============================
Birden fazla strateji/portfoyun fiyat/getiri serilerini karsilastirmak
icin yardimci fonksiyonlar: ortak bir baslangic degerine indirgeme
(`rebase`, karsilastirmali grafik icin) ve ozet performans istatistik
tablosu (`compare_strategies`).

TUTARLILIK icin, Sharpe ve max drawdown gibi mevcut metrikler burada
YENIDEN yazilmaz - `quanttro.backtest.metrics` modulundeki fonksiyonlar
dogrudan import edilip kullanilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quanttro.backtest.metrics import max_drawdown, sharpe_ratio

TRADING_DAYS_PER_YEAR = 252


def rebase(prices: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    """
    Bir fiyat DataFrame'inin HER KOLONUNU, kendi ilk (ilk satir) degerine
    gore ayni baslangic degerine (`base_value`) indirger::

        rebased[kolon] = prices[kolon] / prices[kolon].iloc[0] * base_value

    Farkli mutlak fiyat seviyelerindeki birden fazla varlik/strateji
    getiri egrisini AYNI eksende, karsilastirilabilir sekilde cizmeden
    once kullanilir (orn. biri 100 TL'den, biri 10.000 TL'den basliyorsa).
    NaN olan ilk deger, o kolonun tamamini NaN yapar (kasitli - rebase
    noktasi tanimsizsa sonuc da tanimsiz olmalidir).
    """
    first_row = prices.iloc[0]
    return prices.div(first_row, axis=1).mul(base_value)


def compare_strategies(
    returns_dict: dict[str, pd.Series],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """
    Birden fazla strateji/portfoyun periyot bazli (basit/aritmetik) getiri
    serisini alir, her biri icin ozet performans istatistik tablosu
    uretir.

    Her strateji icin hesaplanan kolonlar:
    - ``total_return``: toplam (kumulatif) getiri, ``(1+r).cumprod().iloc[-1] - 1``.
    - ``annualized_return``: yillliklandirilmis getiri (CAGR),
      ``total_growth ** (periods_per_year / n) - 1``.
    - ``annualized_volatility``: yillliklandirilmis volatilite,
      ``r.std(ddof=1) * sqrt(periods_per_year)``.
    - ``sharpe_ratio``: `quanttro.backtest.metrics.sharpe_ratio` ile
      (TUTARLILIK icin buradan yeniden hesaplanmaz, dogrudan kullanilir).
    - ``max_drawdown``: `quanttro.backtest.metrics.max_drawdown` ile,
      kumulatif getiri egrisi uzerinden (negatif deger, orn. -0.23 -> %23).

    Donus: satirlar = strateji adi (``returns_dict`` anahtarlari), kolonlar
    = yukaridaki 5 metrik olan bir DataFrame. Gecerli (NaN'siz) gozlem
    sayisi 2'den az olan bir strateji icin tum metrikler NaN olur.
    """
    rows: dict[str, dict[str, float]] = {}

    for name, returns in returns_dict.items():
        r = pd.Series(returns, dtype=float).dropna()
        n = len(r)

        if n < 2:
            rows[name] = {
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "annualized_volatility": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
            }
            continue

        cumulative = (1.0 + r).cumprod()
        total_growth = float(cumulative.iloc[-1])
        total_return = total_growth - 1.0
        annualized_return = (
            total_growth ** (periods_per_year / n) - 1.0 if total_growth > 0 else float("nan")
        )
        annualized_volatility = float(r.std(ddof=1) * np.sqrt(periods_per_year))

        rows[name] = {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio(r, periods_per_year=periods_per_year, risk_free=risk_free),
            "max_drawdown": max_drawdown(cumulative),
        }

    return pd.DataFrame.from_dict(rows, orient="index")[
        ["total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]
    ]
