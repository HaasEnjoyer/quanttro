"""Monte Carlo simulasyonu ile getiri serisi saglamlik testi.

Bir backtest'in gerceklesen tek bir getiri yoluna (path) dayanmasi, o
sonucun ne kadar "sans eseri" olabilecegini gostermez. Bu modul,
gozlemlenen getiri serisinden coklu alternatif yol uretir (resampling)
ve gercek sonucu bu dagilimin icinde konumlandirmaya yarar:

- ``"shuffle"``: getirilerin SIRASINI rastgele karistirir (trade-order
  etkisini test eder - ayni getiriler farkli bir sirada gelsin, sonuc
  degisir mi?). Getiriler arasi seri korelasyonu (autocorrelation) YOK
  EDER.
- ``"iid_bootstrap"``: getirilerden, BAGIMSIZ VE AYNI DAGILIMDAN
  geldigi varsayimiyla, yerine-koyarak (with replacement) orneklem
  cekilir. ``"shuffle"`` gibi seri korelasyonu yok eder, ustune bazi
  getirileri tekrar, bazilarini hic kullanmaz.
- ``"block_bootstrap"``: ardisik ``block_size`` uzunlugunda bloklar
  (orijinal sirali alt-diziler) rastgele secilip arka arkaya eklenerek
  orijinal uzunluga ulasilir. Blok ici seri korelasyonu (orn. volatilite
  kumelenmesi, ardisik getiriler arasi bagimlilik) KORUNUR - bu yuzden
  ``"shuffle"``/``"iid_bootstrap"``'a gore daha gercekci bir alternatif
  senaryo uretir. Blok, orijinal serinin sonuna gelindiginde BASA
  SARARAK (dairesel/circular) secilir - boylece serinin HER noktasi
  gecerli bir blok baslangicidir ve her blok tam ``block_size``
  uzunlugunda kalir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Getiri serisi uzerinde Monte Carlo yeniden-orneklemesi.

    Parametreler
    ------------
    method:
        ``"block_bootstrap"`` (varsayilan), ``"shuffle"`` ya da
        ``"iid_bootstrap"``.
    block_size:
        Sadece ``method="block_bootstrap"`` icin: her bloğun uzunlugu
        (periyot sayisi).
    random_state:
        Tekrarlanabilirlik icin rastgelelik tohumu (None = her
        cagrimda farkli sonuc).
    """

    _METHODS = ("block_bootstrap", "shuffle", "iid_bootstrap")

    def __init__(
        self,
        method: str = "block_bootstrap",
        block_size: int = 20,
        random_state: int | None = None,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"method {self._METHODS} icinden biri olmalidir")
        if block_size < 1:
            raise ValueError("block_size en az 1 olmalidir")

        self.method = method
        self.block_size = block_size
        self.random_state = random_state

    def _block_bootstrap_sample(self, values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        n_obs = len(values)
        pieces: list[np.ndarray] = []
        total = 0
        while total < n:
            start = int(rng.integers(0, n_obs))
            # Dairesel (circular) secim: start + block_size, n_obs'u asarsa
            # bastan devam eder - boylece serinin SONUNA yakin baslangic
            # noktalari da gecerli olur ve her blok tam block_size kalir.
            positions = (np.arange(start, start + self.block_size)) % n_obs
            pieces.append(values[positions])
            total += self.block_size
        return np.concatenate(pieces)[:n]

    def run(self, returns: pd.Series, n_sims: int = 1000) -> pd.DataFrame:
        """``n_sims`` adet simule edilmis kumulatif getiri (equity) egrisi
        uretir.

        Her sutun bir simulasyondur (0..n_sims-1 ile adlandirilir); her
        egri, orijinal ``returns`` ile AYNI UZUNLUKTADIR ve ``(1 + r)``
        carpimsal kumulatif urunu olarak (1.0'dan baslayarak) kurulur -
        :func:`quanttro.backtest.metrics.max_drawdown` ile dogrudan
        uyumlu bir "equity curve" formatidir. Satir indeksi, orijinal
        ``returns.index`` ile aynidir (zaman eksenini gorsellestirmek
        icin kullanilabilir; ``"shuffle"``/``"iid_bootstrap"`` modunda
        bu sadece bir eksen etiketidir, kronolojik anlami yoktur).

        ``n_sims`` en az 1 olmalidir; ``returns`` en az 1 gecerli
        (NaN olmayan) deger icermelidir.
        """
        if n_sims < 1:
            raise ValueError("n_sims en az 1 olmalidir")

        clean = pd.Series(returns).dropna()
        n = len(clean)
        if n == 0:
            raise ValueError("returns en az bir gecerli deger icermelidir")

        values = clean.to_numpy(dtype=float)
        rng = np.random.default_rng(self.random_state)

        columns: dict[int, np.ndarray] = {}
        for i in range(n_sims):
            if self.method == "shuffle":
                sim_returns = rng.permutation(values)
            elif self.method == "iid_bootstrap":
                sim_returns = rng.choice(values, size=n, replace=True)
            else:  # block_bootstrap
                sim_returns = self._block_bootstrap_sample(values, n, rng)

            columns[i] = np.cumprod(1.0 + sim_returns)

        return pd.DataFrame(columns, index=clean.index)

    def summary(self, simulated_curves: pd.DataFrame) -> dict:
        """Simulasyon sonuclarini ozetler.

        Donus
        -----
        ``{"percentile_bands": pd.DataFrame, "max_drawdown_mean": float,
        "max_drawdown_median": float, "max_drawdown_p95_worst_case": float}``

        - ``percentile_bands``: her zaman noktasinda, TUM simulasyonlar
          arasindaki 5/25/50/75/95. persentil degerlerini iceren bir
          DataFrame (fan-chart cizimi icin uygun; index = ``simulated_curves``
          ile ayni, kolonlar = ``p5, p25, p50, p75, p95``).
        - ``max_drawdown_mean`` / ``max_drawdown_median``: her simulasyonun
          kendi maksimum drawdown'u (NEGATIF deger, orn. -0.23) uzerinden
          ortalama/medyan.
        - ``max_drawdown_p95_worst_case``: drawdown dagiliminin EN KOTU
          %5'lik kuyruguna karsilik gelen deger - drawdown negatif
          oldugundan bu, dagilimin (ham, isaretli) 5. persentilidir
          (yani simulasyonlarin %95'inde maksimum drawdown bu degerden
          DAHA HAFIF/DAHA AZ NEGATIF cikar - standart "%95 guven
          seviyesinde en kotu durum" / VaR yorumu).
        """
        if simulated_curves.shape[1] == 0:
            raise ValueError("simulated_curves en az bir simulasyon (kolon) icermelidir")

        quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        bands = simulated_curves.quantile(quantile_levels, axis=1).T
        bands.columns = ["p5", "p25", "p50", "p75", "p95"]

        drawdowns = simulated_curves.apply(_max_drawdown, axis=0)

        return {
            "percentile_bands": bands,
            "max_drawdown_mean": float(drawdowns.mean()),
            "max_drawdown_median": float(drawdowns.median()),
            "max_drawdown_p95_worst_case": float(drawdowns.quantile(0.05)),
        }


def _max_drawdown(curve: pd.Series) -> float:
    """Tek bir kumulatif (carpimsal) getiri egrisinin maksimum drawdown'u.

    ``quanttro.backtest.metrics.max_drawdown`` ile ayni formul
    (``min(curve / running_max - 1)``) - bu modulu, es zamanli
    gelistirilen ``metrics.py``'ye bagimli kilmamak icin burada
    bagimsiz/kucuk bir kopyasi tutulur.
    """
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())
