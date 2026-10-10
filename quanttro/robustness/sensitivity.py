"""Parametre hassasiyeti (sensitivity) analizi.

Bir stratejinin "en iyi" parametre kombinasyonunun tek bir noktada
(nokta tahmini) iyi performans gostermesi yeterli degildir - o nokta
komsulugunda (kucuk parametre degisikliklerinde) da performans stabil
kalmali (``"plateau"``/plato), aksi halde o sonuc kirilgan (``"cliff"``/
ucurum) ve buyuk olasilikla overfitting urunudur. Bu modul, her
parametreyi TEK TEK (digerleri sabit tutularak, "one-at-a-time"
duyarlilik analizi) tarayip performans egrisinin plato mu ucurum mu
oldugunu siniflandirir.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def parameter_sensitivity(
    backtest_func: Callable[..., float],
    param_grid: dict[str, list],
    base_params: dict,
) -> pd.DataFrame:
    """Her parametreyi, digerleri ``base_params``'ta SABIT tutularak, tek
    tek ``param_grid``'deki degerlerde tarar.

    Parametreler
    ------------
    backtest_func:
        ``backtest_func(**params) -> float`` imzasinda bir fonksiyon;
        tek bir performans skoru (orn. Sharpe orani) dondurmesi
        beklenir.
    param_grid:
        ``{parametre_adi: [deger1, deger2, ...]}`` seklinde, taranacak
        parametre-deger listeleri.
    base_params:
        Taranmayan parametrelerin sabit tutulacagi degerleri iceren
        sozluk (taranan parametrenin kendisi her adimda bu sozlukteki
        karsiligi UZERINE YAZILARAK gecilir).

    Donus
    -----
    Satirlari ``(param_name, param_value)`` ikili (MultiIndex,
    isimleri ``"param_name"``/``"param_value"``) olan, tek kolonlu
    (``"score"``) uzun-format bir DataFrame.
    """
    index_tuples: list[tuple[str, object]] = []
    scores: list[float] = []

    for param_name, values in param_grid.items():
        for value in values:
            params = dict(base_params)
            params[param_name] = value
            score = backtest_func(**params)
            index_tuples.append((param_name, value))
            scores.append(score)

    index = pd.MultiIndex.from_tuples(index_tuples, names=["param_name", "param_value"])
    return pd.DataFrame({"score": scores}, index=index)


def classify_sensitivity(
    sensitivity_df: pd.DataFrame,
    param_name: str,
    plateau_threshold: float = 0.1,
) -> str:
    """Bir parametrenin performans egrisini "plateau" (dayanikli) ya da
    "cliff" (kirilgan) olarak siniflandirir.

    Siniflandirma, o parametreye ait performans skorlarinin
    DEGISIM KATSAYISINA (coefficient of variation, ``|std / mean|``,
    ornek/``ddof=1`` standart sapmasiyla) gore yapilir: katsayi
    ``plateau_threshold``'u asarsa ``"cliff"``, asmazsa ``"plateau"``
    dondurulur.

    ``sensitivity_df``, :func:`parameter_sensitivity` ciktisiyla ayni
    formatta (MultiIndex ``param_name``/``param_value``, kolon
    ``"score"``) olmalidir.
    """
    try:
        subset = sensitivity_df.xs(param_name, level="param_name")["score"]
    except KeyError as exc:
        raise ValueError(f"'{param_name}' sensitivity_df icinde bulunamadi") from exc

    subset = subset.dropna()
    if len(subset) < 2:
        raise ValueError(f"'{param_name}' icin en az 2 gecerli skor gereklidir")

    mean = float(subset.mean())
    std = float(subset.std(ddof=1))

    if np.isclose(mean, 0.0, atol=1e-12):
        # Ortalama ~0 ise oransal (std/mean) karsilastirma tanimsizdir;
        # std de ~0 ise (sabit ve sifira yakin skor) plato, degilse
        # (buyuk degisim, sifir etrafinda) ucurum kabul edilir.
        return "plateau" if np.isclose(std, 0.0, atol=1e-12) else "cliff"

    coefficient_of_variation = abs(std / mean)
    return "cliff" if coefficient_of_variation > plateau_threshold else "plateau"
