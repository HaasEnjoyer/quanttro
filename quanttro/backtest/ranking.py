"""Kesitsel siralama (cross-sectional ranking) degerlendirme metrikleri.

train_model.py icindeki ``run_ranking`` fonksiyonunda tek bir yerde
hesaplanan Information Coefficient (IC) ve tertile long-short spread
mantiginin, bagimsiz/yeniden kullanilabilir fonksiyonlara ayristirilmis
hali:

- :func:`information_coefficient`: tek periyotluk IC (Spearman varsayilan)
- :func:`tertile_spread`: ust-alt grup spread'i, grup sayisi parametrik
  (train_model.py'da sabit 3'tu, artik quintile vb. de yapilabilir)
- :func:`rolling_ic`: gun gun hesaplanan kesitsel IC'nin hareketli
  ortalamasi (zaman icindeki IC kararliligini/azalmasini gormek icin)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    predicted: pd.Series,
    actual: pd.Series,
    method: str = "spearman",
) -> float:
    """Tek bir periyot/kesit icin Information Coefficient (IC): tahmin
    edilen siralama ile gerceklesen getiri arasindaki korelasyon.

    ``method``: "spearman" (siralama korelasyonu, varsayilan - train_model.py
    ile aynı), "pearson" ya da "kendall" (``pandas.Series.corr``'un
    destekledigi yontemler). Gecerli (NaN olmayan, ortak indeksli) nokta
    sayisi 3'ten azsa NaN dondurur.
    """
    pred, act = predicted.align(actual, join="inner")
    valid = pred.notna() & act.notna()
    pred, act = pred[valid], act[valid]
    if len(pred) < 3:
        return float("nan")
    return float(pred.corr(act, method=method))


def tertile_spread(
    predicted: pd.Series,
    actual: pd.Series,
    n_groups: int = 3,
) -> float:
    """Ust-alt grup long-short spread'i: tahmine gore siralanan evrenin en
    yuksek tahminli grubunun ortalama gerceklesen getirisi EKSI en dusuk
    tahminli grubunun ortalama gerceklesen getirisi.

    ``n_groups`` ile tertile (3, varsayilan - train_model.py'nin sabit
    degeri), quintile (5) vb. herhangi bir grup sayisi kullanilabilir.
    Grup buyuklugu ``len(evren) // n_groups`` (train_model.py'daki
    ``n_tercile`` mantiginin genellestirilmis hali) olarak alinir, en az 1.
    """
    if n_groups < 2:
        raise ValueError("n_groups en az 2 olmalidir")
    pred, act = predicted.align(actual, join="inner")
    valid = pred.notna() & act.notna()
    pred, act = pred[valid], act[valid]
    if len(pred) < n_groups:
        return float("nan")

    ranked = pd.DataFrame({"predicted": pred, "actual": act}).sort_values("predicted")
    group_size = max(len(ranked) // n_groups, 1)
    bottom = ranked["actual"].iloc[:group_size].mean()
    top = ranked["actual"].iloc[-group_size:].mean()
    return float(top - bottom)


def rolling_ic(
    predicted_by_date: pd.DataFrame,
    actual_by_date: pd.DataFrame,
    window: int = 60,
    method: str = "spearman",
) -> pd.Series:
    """Her tarihte kesitsel (cross-sectional) IC hesaplayip, bu IC
    serisinin ``window`` periyotluk hareketli ortalamasini dondurur.

    ``predicted_by_date`` ve ``actual_by_date``: satirlari tarih,
    sutunlari hisse/sembol olan (wide-format) DataFrame'ler - her hucre,
    o tarihte o hissenin tahmini/gerceklesen degeridir. Iki DataFrame
    indeks (tarih) bazinda ic kesisimlerine (``inner`` join) gore
    hizalanir; bir tarihte 3'ten az gecerli (ortak, NaN olmayan) sembol
    varsa o tarihin gunluk IC'si NaN kabul edilir.
    """
    predicted_by_date, actual_by_date = predicted_by_date.align(actual_by_date, join="inner")

    daily_ic = {}
    for dt in predicted_by_date.index:
        pred_row = predicted_by_date.loc[dt]
        act_row = actual_by_date.loc[dt]
        valid = pred_row.notna() & act_row.notna()
        if valid.sum() < 3:
            daily_ic[dt] = np.nan
            continue
        daily_ic[dt] = pred_row[valid].corr(act_row[valid], method=method)

    ic_series = pd.Series(daily_ic, name="ic").sort_index()
    min_periods = max(2, window // 3)
    return ic_series.rolling(window=window, min_periods=min_periods).mean()
