"""Probability of Backtest Overfitting (PBO).

Referans: Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J.
(2017). "The Probability of Backtest Overfitting". Journal of Computational
Finance. Yontemin temeli CSCV (Combinatorially Symmetric Cross-Validation)
prosedurudur.

Problem: bir arastirmaci N farkli strateji/parametre kombinasyonu denediyse
ve bunlardan "en iyi" gorunen (en yuksek in-sample performansli) stratejiyi
secip raporladiysa, bu secimin kendisi bir coklu-test (multiple testing)
sorunudur - secilen strateji buyuk olasilikla gercek bir beceriden (skill)
degil, saf sanstan (gurultunun en iyi gorundugu taraf) kaynaklanabilir.
PBO, "in-sample'de en iyi secilen stratejinin out-of-sample'de de iyi
performans gostermeme olasiligi"ni tahmin eder.

CSCV prosedurunun ozeti (bkz. :func:`probability_of_backtest_overfitting`):

1) T periyot x N strateji performans matrisi, zaman eksininde ``S``
   (``n_partitions``) esit parcaya bolunur.
2) Bu S parcanin YARISI (S/2) ile olusturulabilecek TUM
   ``C(S, S/2)`` kombinasyonu denenir; her kombinasyonda secilen S/2
   parca IS (in-sample), kalan S/2 parca OOS (out-of-sample) olur.
3) Her kombinasyonda: IS verisinde en iyi performansi gosteren strateji
   bulunur (n*); bu stratejinin OOS verisindeki BAGIL SIRASI (rank,
   N stratejl arasinda, [0, 1] araligina normalize) hesaplanir.
4) Bu bagil sira, bir "logit" donusumunden (``ln(rank / (1 - rank))``)
   gecirilir. Logit <= 0 ise, n* stratejisi OOS'ta medyanin (ya da
   altinda) kalmistir - yani in-sample'deki ustunlugu out-of-sample'de
   KORUNMAMISTIR (overfitting isareti).
5) PBO = logit <= 0 olan kombinasyonlarin oranidir.

Rastgele (gercek sinyali olmayan) denemelerde, IS'te "en iyi" secilen
strateji OOS'ta sans eseri herhangi bir sirada cikabilecegi icin PBO
~0.5'e yakin (yuksek) cikmasi beklenir. Tutarli, gercek bir sinyali olan
bir strateji ise IS'te de OOS'te de sistematik olarak iyi performans
gosterecegi icin PBO DUSUK cikar.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _column_performance(returns: np.ndarray) -> np.ndarray:
    """Her kolon (strateji/deneme) icin basit performans skoru (mean/std,
    yani yillliklandirilmamis bir Sharpe benzeri olcum).

    CSCV prosedurunde sadece stratejiler ARASI BAGIL siralama onemlidir;
    bu yuzden yillliklandirma ya da risk-free duzeltmesi gereksizdir.
    Std ~0 (sabit/sabit-benzeri getiri serisi) durumunda, bolme
    kararsizligini onlemek icin dogrudan ortalama kullanilir.
    """
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    safe_std = np.where(std > 1e-12, std, 1.0)
    return np.where(std > 1e-12, mean / safe_std, mean)


def probability_of_backtest_overfitting(
    trial_results: pd.DataFrame,
    n_partitions: int = 10,
) -> dict:
    """PBO'yu CSCV prosedurune gore hesaplar (Bailey, Borwein, Lopez de
    Prado & Zhu, 2017).

    Parametreler
    ------------
    trial_results:
        Satirlar = zaman periyotlari (T adet), kolonlar = farkli
        strateji/parametre denemeleri (N adet, N >= 2 olmalidir),
        degerler = o periyottaki getiri/performans. Eksik (NaN) deger
        icermemelidir.
    n_partitions:
        Zaman eksininin bolunecegi parca sayisi (``S``); CIFT ve en az 2
        olmalidir (IS/OOS'un S/2'ser parcaya esit bolunmesi icin).
        Varsayilan 10 -> ``C(10, 5) = 252`` kombinasyon (hesaplama
        maliyeti makul kalir). Makaledeki varsayilan S=16
        (``C(16,8)=12870``) daha hassas ama daha maliyetlidir.

    Donus
    -----
    ``{"pbo": float, "n_combinations_tested": int, "logits": list[float]}``

    - ``pbo``: [0, 1] araliginda, logit <= 0 olan kombinasyonlarin orani.
    - ``n_combinations_tested``: denenen ``C(n_partitions, n_partitions/2)``
      kombinasyon sayisi.
    - ``logits``: her kombinasyon icin hesaplanan logit degerleri (dagilim
      incelemesi/teshis icin).
    """
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions en az 2 ve CIFT olmalidir (S/2 IS/OOS bolunmesi icin)")

    values = trial_results.to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("trial_results iki boyutlu (T x N) olmalidir")
    n_periods, n_trials = values.shape
    if n_trials < 2:
        raise ValueError("En az 2 strateji/deneme (kolon) gereklidir")
    if n_periods < n_partitions:
        raise ValueError(
            f"Zaman periyodu sayisi ({n_periods}), n_partitions ({n_partitions}) kadar olmalidir"
        )
    if np.isnan(values).any():
        raise ValueError("trial_results NaN deger icermemelidir")

    groups = np.array_split(np.arange(n_periods), n_partitions)
    half = n_partitions // 2

    logits: list[float] = []
    below_median_count = 0

    for is_group_idx in combinations(range(n_partitions), half):
        is_set = set(is_group_idx)
        oos_group_idx = [g for g in range(n_partitions) if g not in is_set]

        is_positions = np.concatenate([groups[g] for g in is_group_idx])
        oos_positions = np.concatenate([groups[g] for g in oos_group_idx])

        is_scores = _column_performance(values[is_positions])
        best_strategy = int(np.argmax(is_scores))

        oos_scores = _column_performance(values[oos_positions])
        # best_strategy'nin OOS performansinin N strateji arasindaki bagil
        # sirasi: ortalama-rank (beraberlikleri adil bolusturur), 1 (en
        # kotu) ile N (en iyi) arasinda; (N+1)'e bolunerek (0, 1) araligina
        # normalize edilir (logit donusumu icin uclarda sifir/sonsuza
        # gitmemesi icin).
        oos_ranks = rankdata(oos_scores, method="average")
        relative_rank = oos_ranks[best_strategy] / (n_trials + 1)
        relative_rank = min(max(relative_rank, 1e-6), 1.0 - 1e-6)

        logit = math.log(relative_rank / (1.0 - relative_rank))
        logits.append(logit)
        if logit <= 0:
            below_median_count += 1

    n_combinations = len(logits)
    pbo = below_median_count / n_combinations

    return {
        "pbo": float(pbo),
        "n_combinations_tested": n_combinations,
        "logits": logits,
    }
