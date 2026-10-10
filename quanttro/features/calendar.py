"""
quanttro.features.calendar
==============================
Takvim/zaman bazli ozellikler.

Bu modul `feature_engineering.py` icindeki day_of_week / is_turn_of_month /
hour_of_day mantigini genellestirir. Ayrica bu arastirma oturumunda
(bkz. `validate_volume_dow_rule.py`) BIST'te GERCEKTEN anlamli cikan
"Pazartesi/Sali + yuksek hacim -> farkli ileri getiri" bulgusunun takvim
ayagini (`is_monday_or_tuesday`) genel bir yardimci fonksiyon olarak ekler;
yuksek hacim kosulu icin `core.volume_zscore` ile birlikte kullanilmasi
beklenir.
"""

from __future__ import annotations

import pandas as pd


def day_of_week_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Haftanin gunu icin hem ham kategori (0=Pazartesi .. 6=Pazar) hem de
    one-hot kodlanmis kolonlar ("dow_0".."dow_6") uretir.

    Tum 7 gun kolonu HER ZAMAN uretilir (veri setinde o gun hic gorulmemis
    olsa bile - orn. sadece is gunleri icin 5/6 asla gorulmez) boylece
    farkli tarih araliklarinda kolon semasi sabit kalir (ML pipeline'lari
    icin onemli).
    """
    dow = pd.Series(index.dayofweek, index=index, name="day_of_week")
    dow_cat = dow.astype(pd.CategoricalDtype(categories=list(range(7))))
    one_hot = pd.get_dummies(dow_cat, prefix="dow").astype(int)
    return pd.concat([dow, one_hot], axis=1)


def turn_of_month_flag(index: pd.DatetimeIndex, n_days: int = 3) -> pd.Series:
    """
    Ay donumu (turn-of-month) flag'i: ayin ilk `n_days` gunu VEYA son
    `n_days` gunu icindeyse 1, degilse 0.

    feature_engineering.py'daki is_turn_of_month mantiginin genellestirilmis
    hali; varsayilan n_days=3 ile orijinal (day_of_month <= 3) |
    (day_of_month >= days_in_month - 2) ifadesiyle AYNI sonucu uretir.
    """
    day_of_month = index.day
    days_in_month = index.days_in_month
    flag = (day_of_month <= n_days) | (day_of_month >= days_in_month - n_days + 1)
    return pd.Series(flag.astype(int), index=index, name="is_turn_of_month")


def hour_of_day_features(index: pd.DatetimeIndex) -> pd.Series:
    """
    Gun ici (intraday) veri icin saat bilgisi (0-23). Gunluk veri icin
    anlamsizdir (sabit tek bir deger doner).
    """
    return pd.Series(index.hour, index=index, name="hour_of_day")


def is_monday_or_tuesday(index: pd.DatetimeIndex) -> pd.Series:
    """
    Pazartesi (0) veya Sali (1) ise 1, degilse 0.

    Bu oturumda walk-forward dogrulamasinda (`validate_volume_dow_rule.py`,
    TARGET_DOWS = {0: Pazartesi, 1: Sali}) bulunan "Pazartesi/Sali + yuksek
    hacim -> farkli ileri getiri" kuralinin takvim ayagidir; anlamli sinyal
    icin `core.volume_zscore` ile (yuksek hacim kosulu) BIRLIKTE
    kullanilmasi beklenir - bu fonksiyon tek basina sadece gun bilgisini
    tasir.
    """
    flag = index.dayofweek.isin([0, 1])
    return pd.Series(flag.astype(int), index=index, name="is_monday_or_tuesday")
