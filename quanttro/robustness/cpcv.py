"""Combinatorial Purged Cross-Validation (CPCV).

Marcos Lopez de Prado'nun "Advances in Financial Machine Learning" (2018)
kitabinin 12. bolumunde tanimladigi yontem. Standart k-fold CV, finansal
zaman serilerinde iki temel soruna yol acar:

1) **Bilgi sizintisi (leakage)**: train ve test setleri birbirine zaman
   icinde bitisik oldugunda, orneklerin etiketleri (orn. N-bar ileri
   getiri) ya da hesaplanmasinda kullanilan kayan-pencere ozellikler
   train/test sinirini asip "sizabilir". ``purge_gap`` ve ``embargo_pct``
   bu sizintiyi sinir boyunca temizler (purge/embargo).
2) **Tek bir train/test yolu**: standart (purged) k-fold CV tek bir
   train/test ayrimi uretir, bu da performans tahmininin varyansini
   (ve dolayisiyla "backtest overfitting" riskini) olcmeyi imkansiz
   kilar. CPCV, veriyi ``n_groups`` esit parcaya bolup, bunlardan
   ``n_test_groups`` kadarini test, kalanini train yaparak TUM
   C(n_groups, n_test_groups) kombinasyonunu dener - boylece coklu,
   birbirinden farkli (ve kismen ortusen) train/test yollari elde
   edilir. Bu yollar daha sonra :mod:`quanttro.robustness.overfitting`
   modulundeki PBO hesabinin da girdisidir.

ONEMLI: CPCV'de test gruplari ZAMAN ICINDE ARDISIK (bitisik) OLMAK
ZORUNDA DEGILDIR - ornegin grup 0 ve grup 3 test, 1-2-4-5 train olabilir.
Bu yuzden purge/embargo, test gruplarinin OLUSTURDUGU TEK BIR BLOGUN
disinda, HER BIR test grubunun KENDI sinirlari etrafinda ayri ayri
uygulanir (bkz. :meth:`CombinatorialPurgedCV.split`).
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation bolme uretici.

    Veriyi ``n_groups`` esit (mumkun oldugunca - ``numpy.array_split``
    kurallarina gore) parcaya boler ve bu gruplardan ``n_test_groups``
    tanesini test, kalanlarini train olarak kullanan TUM
    ``C(n_groups, n_test_groups)`` kombinasyonunu uretir.

    Parametreler
    ------------
    n_groups:
        Verinin bolunecegi esit parca (grup) sayisi. En az 2 olmalidir.
    n_test_groups:
        Her kombinasyonda test olarak kullanilacak grup sayisi
        (``1 <= n_test_groups < n_groups``).
    purge_gap:
        Her test grubunun basindan ve sonundan itibaren, egitimden
        cikarilacak BITISIK (pozisyon bazli) ornek sayisi. Train
        orneklerinin etiket/ozellik pencerelerinin test sinirini asip
        sizinti yaratmasini onlemek icindir; her test grubunun sinirinda
        BAGIMSIZ olarak uygulanir (test gruplari ardisik olmak zorunda
        olmadigindan).
    embargo_pct:
        Her test grubunun SONUNDAN itibaren, egitimden EK OLARAK
        cikarilacak orneklerin, toplam veri uzunluguna oran olarak
        buyuklugu (``0.0 <= embargo_pct < 1.0``). Sadece test'in
        ARDINDAN gelen (ileri yonde) orneklere uygulanir - seri
        korelasyon nedeniyle test sonrasi train orneklerinin test
        donemiyle hala iliskili olabilecegi ekstra bir guvenlik payidir.
        Gercek ornek sayisina ``round(len(index) * embargo_pct)`` ile
        cevrilir.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups en az 2 olmalidir")
        if not (1 <= n_test_groups < n_groups):
            raise ValueError("n_test_groups, 1 <= n_test_groups < n_groups kosulunu saglamalidir")
        if purge_gap < 0:
            raise ValueError("purge_gap negatif olamaz")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError("embargo_pct [0.0, 1.0) araliginda olmalidir")

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def n_splits(self) -> int:
        """Toplam kombinasyon sayisi: ``C(n_groups, n_test_groups)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    def split(self, index: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """``index`` uzerinde tum CPCV kombinasyonlarini uretir.

        Her kombinasyon icin ``(train_positions, test_positions)``
        dondurur - bunlar ZAMAN DAMGASI DEGIL, ``index`` icindeki
        POZISYONLARDIR (integer, sirali), yani dogrudan
        ``df.iloc[train_positions]`` / ``df.iloc[test_positions]`` ile
        kullanilabilir. Kombinasyonlar, ``itertools.combinations`` sirasiyla
        (leksikografik) uretilir.
        """
        n = len(index)
        if n < self.n_groups:
            raise ValueError(f"Veri uzunlugu ({n}), n_groups ({self.n_groups}) kadar olmalidir")

        group_positions = np.array_split(np.arange(n), self.n_groups)
        embargo_size = int(round(n * self.embargo_pct))

        for test_group_idx in combinations(range(self.n_groups), self.n_test_groups):
            test_positions = np.sort(
                np.concatenate([group_positions[g] for g in test_group_idx])
            )

            excluded: set[int] = set(test_positions.tolist())

            # Purge + embargo: HER test grubunun kendi sinirlari etrafinda
            # ayri ayri uygulanir (test gruplari ardisik olmak zorunda
            # degil - bkz. modul docstring'i).
            for g in test_group_idx:
                block = group_positions[g]
                if len(block) == 0:
                    continue
                start, end = int(block[0]), int(block[-1])
                purge_before = range(max(0, start - self.purge_gap), start)
                purge_after = range(end + 1, min(n, end + 1 + self.purge_gap + embargo_size))
                excluded.update(purge_before)
                excluded.update(purge_after)

            train_positions = np.array(sorted(set(range(n)) - excluded), dtype=int)

            yield train_positions, test_positions
