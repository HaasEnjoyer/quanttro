"""
quanttro.portfolio.optimize
================================
Ortalama-varyans (mean-variance) portfoy optimizasyonu: Sharpe oranini
maksimize eden ve minimum-volatilite portfoyleri, hedef bir volatilite
altinda beklenen getiriyi maksimize eden "efficient risk" portfoyu, ve
Ledoit-Wolf kovaryans shrinkage tahmini.

Agir bir konveks-optimizasyon bagimliligi (cvxpy) KULLANILMAZ - tum
optimizasyonlar `scipy.optimize.minimize` (SLSQP, esitlik/esitsizlik
kisitlariyla) ile cozulur. PyPortfolioOpt'un sundugu temel mean-variance/
efficient-frontier islevselliginin basitlestirilmis ama dogru calisan bir
alt kumesidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _align(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """`expected_returns`'un sembol sirasini referans alarak `cov_matrix`'i hizalar."""
    symbols = expected_returns.index
    cov = cov_matrix.reindex(index=symbols, columns=symbols)
    if cov.isna().any().any():
        missing = cov.columns[cov.isna().any()].tolist()
        raise ValueError(
            f"cov_matrix, expected_returns'teki tum sembolleri icermiyor (eksik: {missing})"
        )
    return expected_returns, cov


def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """w^T * Cov * w - portfoy varyansi."""
    return float(weights @ cov @ weights)


def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free: float = 0.0,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.Series:
    """
    Ortalama-varyans optimizasyonu (`scipy.optimize.minimize`, method="SLSQP").

    `objective`:
    - ``"max_sharpe"``: Sharpe oranini ``(w.mu - risk_free) / sqrt(w.Cov.w)``
      maksimize eden agirliklari bulur (pratikte NEGATIF Sharpe minimize
      edilir, SLSQP sadece minimizasyon yapar).
    - ``"min_volatility"``: beklenen getiriyi gormezden gelip, portfoy
      varyansini (``w.Cov.w``) minimize eden agirliklari bulur.

    Her iki durumda da kisitlar aynidir: ``sum(w) == 1`` ve her agirlik
    ``weight_bounds`` araliginda (varsayilan: long-only, ``[0, 1]`` -
    acik pozisyon/kaldirac yok).

    Donus: `expected_returns` ile ayni sira/index'e sahip, toplami 1 olan
    agirlik Series'i (kucuk sayisal sapmalari gidermek icin sonuc toplama
    gore yeniden normalize edilir).
    """
    expected_returns, cov_matrix = _align(expected_returns, cov_matrix)
    symbols = expected_returns.index
    n = len(symbols)
    if n == 0:
        raise ValueError("expected_returns bos olamaz")

    mu = expected_returns.to_numpy(dtype=float)
    cov = cov_matrix.to_numpy(dtype=float)
    bounds = tuple(weight_bounds for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    w0 = np.full(n, 1.0 / n)

    if objective == "max_sharpe":
        def neg_sharpe(w: np.ndarray) -> float:
            port_return = float(w @ mu)
            # varyans icin kucuk bir taban (1e-18): sifira yakin volatilitede
            # (dejenere/sabit getirili varliklarda) bolme hatasi olusmasin.
            port_vol = np.sqrt(max(_portfolio_variance(w, cov), 1e-18))
            return -(port_return - risk_free) / port_vol

        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "min_volatility":
        result = minimize(
            lambda w: _portfolio_variance(w, cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
    else:
        raise ValueError(
            f"Taninmayan objective: {objective!r} (bekleniyor: 'max_sharpe' veya 'min_volatility')"
        )

    if not result.success:
        raise RuntimeError(f"mean_variance_optimize yakinsamadi ({objective}): {result.message}")

    weights = result.x / result.x.sum()
    return pd.Series(weights, index=symbols, name="weight")


def efficient_risk(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_volatility: float,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.Series:
    """
    Belirtilen `target_volatility`'ye ESIT VEYA ONUN ALTINDA riskte,
    beklenen getiriyi maksimize eden agirliklari bulur (SLSQP, esitsizlik
    kisitiyla).

    Kisitlar:
    - ``sum(w) == 1``
    - her agirlik ``weight_bounds`` araliginda
    - ``w.Cov.w <= target_volatility**2`` (karekok yerine varyans uzerinden
      ifade edilir - sayisal olarak daha kararli ve turevi her yerde
      tanimlidir)

    `target_volatility`, verilen `weight_bounds` kisitlari altinda
    ERISILEMEZ ise (yani minimum mumkun portfoy volatilitesi hedefin
    USTUNDEYSE) ``ValueError`` firlatilir.
    """
    if target_volatility <= 0:
        raise ValueError("target_volatility pozitif olmalidir")

    expected_returns, cov_matrix = _align(expected_returns, cov_matrix)
    symbols = expected_returns.index
    mu = expected_returns.to_numpy(dtype=float)
    cov = cov_matrix.to_numpy(dtype=float)
    bounds = tuple(weight_bounds for _ in range(len(symbols)))

    # erisilebilirlik kontrolu: minimum-volatilite portfoyunun riski hedefin
    # ustundeyse, bu kisitlar altinda hedef volatilite hic erisilemezdir.
    min_vol_weights_series = mean_variance_optimize(
        expected_returns, cov_matrix, objective="min_volatility", weight_bounds=weight_bounds
    )
    min_vol_weights = min_vol_weights_series.to_numpy()
    min_possible_vol = float(np.sqrt(_portfolio_variance(min_vol_weights, cov)))
    if min_possible_vol > target_volatility + 1e-9:
        raise ValueError(
            f"target_volatility ({target_volatility:.4f}) bu agirlik kisitlari altinda "
            f"erisilemez; bu varlik kumesi icin minimum mumkun volatilite: "
            f"{min_possible_vol:.4f}"
        )

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: target_volatility ** 2 - _portfolio_variance(w, cov)},
    )

    result = minimize(
        lambda w: -(w @ mu),
        min_vol_weights,  # fizibil (feasible) bir baslangic noktasi - yakinsamayi kolaylastirir
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"efficient_risk yakinsamadi: {result.message}")

    weights = result.x / result.x.sum()
    return pd.Series(weights, index=symbols, name="weight")


def covariance_shrinkage(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    """
    Kovaryans matrisi tahmini.

    `method="ledoit_wolf"` (varsayilan): Ledoit-Wolf shrinkage
    (`sklearn.covariance.LedoitWolf`) - ornek (sample) kovaryansi, kosegen
    (diagonal) bir hedefe dogru optimal bir oranda "shrink" ederek tahmin
    hatasini azaltir; ozellikle varlik sayisi gozlem sayisina yakin/fazla
    oldugunda ornek kovaryanstan daha kararli/iyi-kosullu (well-conditioned)
    bir tahmin verir.

    `method="sample"`: duz ornek kovaryans (``returns.cov()``), shrinkage
    UYGULANMADAN - karsilastirma/referans amaclidir.

    NaN iceren satirlar hesaplamadan once `dropna` ile atilir. Donus:
    sembol x sembol (kare, simetrik) kovaryans DataFrame'i.
    """
    clean = returns.dropna(how="any")
    if len(clean) < 2:
        raise ValueError("covariance_shrinkage icin en az 2 gecerli (NaN'siz) gozlem gerekir")

    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(clean.to_numpy(dtype=float))
        cov = estimator.covariance_
    elif method == "sample":
        return clean.cov()
    else:
        raise ValueError(f"Taninmayan method: {method!r} (bekleniyor: 'ledoit_wolf' veya 'sample')")

    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)
