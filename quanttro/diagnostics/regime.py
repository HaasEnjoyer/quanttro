"""Rejim kirilmasi (regime break) tespiti icin genel araclar.

analyze_weekly_volume.py'deki walk-forward analizinde bulunan somut ornek: "genel_fark"
kolonunun ilk 4 fold'da hep negatif, son fold'da (2022-06-10 - 2026-09-25) pozitife
donmesi. Bu modul bu tur "son fold onceki fold'lara gore isaret degistirdi mi" sorusunu
ve iki serinin korelasyonunun zaman icinde ne kadar stabil oldugunu genel olarak
degerlendirmek icin yazildi.

Onemli ders (bkz. docs/KNOWN_PITFALLS.md): walk-forward fold'lardan tek bir "ortalama"
sayiya indirgemek, altta yatan rejim degisimini saklayabilir. Bu fonksiyonlar o
degisimi gorunur kilmayi hedefler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def detect_regime_break(
    fold_results: pd.DataFrame,
    value_col: str,
    sign_flip_threshold: int = 1,
) -> dict:
    """Walk-forward fold sonuclarinda son fold(lar)in isaret degistirip degistirmedigini tespit eder.

    `fold_results`, ornegin analyze_weekly_volume.walk_forward_regime_check() ya da
    validate_volume_dow_rule.run() ciktisi gibi, her satirin bir walk-forward fold'una
    karsilik geldigi ve zaman sirasiyla siralanmis bir tablo olmali.

    Parametreler
    ----------
    fold_results : pd.DataFrame
        Her satir bir fold; zaman sirasina gore siralanmis olmali (ilk satir en eski fold).
    value_col : str
        Incelenecek metrik kolonu (ornek: "genel_fark", "fark", "momentum_fark").
    sign_flip_threshold : int
        Son kac fold "recent" grubu olarak ayrilip onceki fold'larin ortalamasiyla
        karsilastirilacak (varsayilan 1 - yani sadece en son fold).

    Donus
    -----
    dict
        {
            "n_folds": toplam fold sayisi,
            "prior_mean": onceki fold'larin value_col ortalamasi,
            "recent_mean": son fold(lar)in value_col ortalamasi,
            "sign_flipped": bool - isaret degisti mi,
            "t_stat": prior vs recent arasindaki t-istatistigi (Welch t-test),
            "p_value": bu farkin p-degeri,
            "warning": isaret degistiyse aciklayici bir mesaj, yoksa None,
        }
    """
    values = fold_results[value_col].dropna().reset_index(drop=True)
    n = len(values)

    base_result = {
        "n_folds": n,
        "prior_mean": float("nan"),
        "recent_mean": float("nan"),
        "sign_flipped": False,
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "warning": None,
    }

    if n < sign_flip_threshold + 2:
        base_result["warning"] = (
            "Rejim kirilmasi degerlendirmesi icin yeterli fold yok "
            f"(en az {sign_flip_threshold + 2} fold gerekli, {n} fold var)."
        )
        return base_result

    recent = values.iloc[n - sign_flip_threshold:]
    prior = values.iloc[: n - sign_flip_threshold]

    prior_mean = float(prior.mean())
    recent_mean = float(recent.mean())
    sign_flipped = bool(
        prior_mean != 0 and recent_mean != 0 and np.sign(prior_mean) != np.sign(recent_mean)
    )

    if len(recent) >= 2 and len(prior) >= 2 and prior.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_ind(recent, prior, equal_var=False)
        t_stat, p_value = float(t_stat), float(p_value)
    elif len(prior) >= 2 and prior.std(ddof=1) > 0:
        # tek bir "recent" degeri varsa: prior dagilimina gore ne kadar uc oldugunu
        # tek-ornek t-istatistigi gibi raporla (resmi p-value hesaplanmaz, NaN birakilir)
        se = prior.std(ddof=1) / np.sqrt(len(prior))
        t_stat = float((recent_mean - prior_mean) / se) if se > 0 else float("nan")
        p_value = float("nan")
    else:
        t_stat, p_value = float("nan"), float("nan")

    warning = None
    if sign_flipped:
        warning = (
            f"UYARI: son {sign_flip_threshold} fold'da '{value_col}' isareti onceki "
            f"fold'lara gore TERS DONDU ({prior_mean:.5f} -> {recent_mean:.5f}). "
            "Tum ornegin tek bir 'ortalama' sonucuna guvenmeyin - rejim kirilmasi olasi "
            "(bkz. docs/KNOWN_PITFALLS.md: 'haftalik hacim-getiri iliskisi rejim-bagimlidir')."
        )

    base_result.update({
        "prior_mean": prior_mean,
        "recent_mean": recent_mean,
        "sign_flipped": sign_flipped,
        "t_stat": t_stat,
        "p_value": p_value,
        "warning": warning,
    })
    return base_result


def rolling_correlation_stability(
    x: pd.Series,
    y: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Iki serinin zaman icindeki yuvarlanan (rolling) korelasyonunu hesaplar.

    Tek bir "tum donem" korelasyon katsayisi, altta yatan iliskinin zaman icinde
    guclenip zayifladigini (ya da tamamen yon degistirdigini) saklayabilir. Bu
    fonksiyon, `window` boyutlu bir pencerede kayan Pearson korelasyonunu dondurur;
    sonucun kendisinin de std()/trend'i alinarak "ne kadar stabil" sorusuna sayisal
    bir cevap uretilebilir (dusen ya da isaret degistiren bir rolling korelasyon,
    rejim kirilmasinin bir on isaretidir).

    Parametreler
    ----------
    x, y : pd.Series
        Karsilastirilacak iki seri (ayni ya da farkli index'li olabilir - once
        kesisen indexlere gore hizalanir).
    window : int
        Yuvarlanan pencere uzunlugu (varsayilan 252 ~ 1 is yili).

    Donus
    -----
    pd.Series
        x ve y'nin kesisen index'i uzerinde, her noktadaki son `window` barlik
        Pearson korelasyonu. Pencere dolmadan NaN doner.
    """
    x_aligned, y_aligned = x.align(y, join="inner")
    return x_aligned.rolling(window).corr(y_aligned)
