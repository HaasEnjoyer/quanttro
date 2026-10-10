"""Walk-forward (ileriye dogru) capraz dogrulama bolme araclari.

train_model.py icindeki `walk_forward_folds` fonksiyonunun genellestirilmis,
yeniden kullanilabilir hali. Orijinaldeki mantik (benzersiz tarihleri
n_folds + 1 parcaya bolup sirayla (train_cutoff, test_start, test_end)
uretmek) korunur, ancak iki zenginlestirme eklenmistir:

- ``purge_gap``: train ile test arasina, test'e en yakin ``purge_gap``
  tarihi egitimden cikararak bosluk (purge) birakir. Ozellikle ileri-bakisli
  (forward-looking) hedef degiskenler (orn. N-bar ileri getiriler)
  kullanilirken train/test sinirindaki bilgi sizintisi riskini azaltir
  (Lopez de Prado'nun "purged cross-validation" fikrinin basitlestirilmis
  hali).
- ``window="rolling"``: sabit genislikte kayan pencere modu. train_model.py
  sadece "expanding" (buyuyen) pencereyi biliyordu; burada egitim penceresi
  sabit sayida blok (``train_window``) genisliginde ileri kayar.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Tek bir walk-forward fold'unun sinirlari.

    Geriye donuk uyumluluk icin ``train_cutoff, test_start, test_end``
    seklinde 3 elemanli bir tuple gibi unpack/index edilebilir (tam olarak
    train_model.py'daki orijinal kullanim sekli). Rolling pencere modunda
    train setinin ALT sinirina da ihtiyac duyulabilecegi icin bu bilgi
    ``train_start`` alaninda ayrica tutulur:

    - Expanding modda ``train_start``, verinin en bastaki tarihidir (train
      seti her zaman baştan baslar) - bu durumda sadece ``index <=
      train_cutoff`` filtresi yeterlidir.
    - Rolling modda train setinin SABIT genislikte olmasi icin ``index >=
      train_start`` filtresinin DE uygulanmasi gerekir; sadece
      ``train_cutoff`` kullanmak (eski expanding-tarzi filtreleme)
      yanlislikla "expanding" davranisi geri getirir.
    """

    train_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start: pd.Timestamp

    def __iter__(self):
        yield self.train_cutoff
        yield self.test_start
        yield self.test_end

    def __getitem__(self, i):
        return (self.train_cutoff, self.test_start, self.test_end)[i]

    def __len__(self) -> int:
        return 3


class WalkForwardSplitter:
    """Zaman serisi / panel verisi icin walk-forward bolme mantigi.

    Parametreler
    ------------
    n_folds:
        Uretilecek fold sayisi. Veri, n_folds + 1 parcaya bolunur; ilk parca
        hic test edilmeden sadece (ilk fold'un) egitim havuzuna girer.
    purge_gap:
        Train ile test arasinda birakilacak benzersiz tarih sayisi (0 =
        bosluk yok, train_model.py'nin orijinal davranisi).
    window:
        "expanding" (varsayilan, buyuyen pencere) ya da "rolling" (sabit
        genislikte kayan pencere).
    train_window:
        Sadece ``window="rolling"`` icin: egitim penceresinin genisligi,
        blok sayisi olarak (1 blok ~= toplam verinin 1/(n_folds+1)'i kadar
        bir zaman dilimi). None verilirse 1 kabul edilir.
    """

    def __init__(
        self,
        n_folds: int = 5,
        purge_gap: int = 0,
        window: str = "expanding",
        train_window: int | None = None,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds en az 1 olmalidir")
        if purge_gap < 0:
            raise ValueError("purge_gap negatif olamaz")
        if window not in ("expanding", "rolling"):
            raise ValueError("window 'expanding' ya da 'rolling' olmalidir")
        if window == "rolling":
            if train_window is None:
                train_window = 1
            if train_window < 1:
                raise ValueError("train_window en az 1 olmalidir")

        self.n_folds = n_folds
        self.purge_gap = purge_gap
        self.window = window
        self.train_window = train_window

    @classmethod
    def expanding_window(cls, n_folds: int = 5, purge_gap: int = 0) -> "WalkForwardSplitter":
        """Expanding-window (buyuyen pencere) modunda splitter uretir: her
        fold'da egitim seti, verinin basindan o ana kadar her seyi icerir
        (train_model.py'nin orijinal davranisi)."""
        return cls(n_folds=n_folds, purge_gap=purge_gap, window="expanding")

    @classmethod
    def rolling_window(
        cls, n_folds: int = 5, purge_gap: int = 0, train_window: int = 1
    ) -> "WalkForwardSplitter":
        """Rolling-window (sabit genislikte kayan pencere) modunda splitter
        uretir: egitim seti her fold'da sabit genislikte (``train_window``
        blok) kalir ve ileri kayar."""
        return cls(n_folds=n_folds, purge_gap=purge_gap, window="rolling", train_window=train_window)

    def split(self, index: pd.DatetimeIndex) -> Iterator[Fold]:
        """``index`` (genelde tekrarli tarihler iceren bir panel verisinin
        DatetimeIndex'i) uzerinde walk-forward fold'larini uretir.

        Her fold icin bir :class:`Fold` dondurur (train_cutoff, test_start,
        test_end olarak unpack edilebilir; rolling modda ayrica
        ``fold.train_start`` de kullanilmalidir - bkz. Fold docstring'i).
        """
        unique_dates = pd.DatetimeIndex(pd.Index(index).unique()).sort_values()
        if len(unique_dates) < self.n_folds + 1:
            raise ValueError(
                f"Fold uretmek icin yetersiz benzersiz tarih: {len(unique_dates)} "
                f"adet var, en az {self.n_folds + 1} gerekli"
            )

        date_values = unique_dates.to_numpy()
        blocks = np.array_split(date_values, self.n_folds + 1)

        for i in range(1, self.n_folds + 1):
            test_block = blocks[i]
            if len(test_block) == 0:
                continue
            test_start, test_end = pd.Timestamp(test_block[0]), pd.Timestamp(test_block[-1])

            if self.window == "expanding":
                train_pool = np.concatenate(blocks[:i])
            else:
                start_idx = max(0, i - self.train_window)
                train_pool = np.concatenate(blocks[start_idx:i])

            if len(train_pool) == 0:
                continue
            train_pool = np.sort(train_pool)

            if self.purge_gap > 0:
                if len(train_pool) <= self.purge_gap:
                    continue
                train_pool = train_pool[: len(train_pool) - self.purge_gap]

            if len(train_pool) == 0:
                continue

            yield Fold(
                train_cutoff=pd.Timestamp(train_pool[-1]),
                test_start=test_start,
                test_end=test_end,
                train_start=pd.Timestamp(train_pool[0]),
            )
