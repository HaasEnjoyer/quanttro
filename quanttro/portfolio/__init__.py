"""
quanttro.portfolio
======================
Portfoy optimizasyonu araclari: ortalama-varyans (mean-variance),
Hierarchical Risk Parity (HRP), surekli agirliklardan tam-sayi hisse/lot
tahsisi (discrete allocation) ve strateji karsilastirma.

PyPortfolioOpt'un sundugu temel islevselligin, agir bir cvxpy bagimliligi
OLMADAN - sadece ``scipy.optimize`` (SLSQP) ve ``scipy.cluster.hierarchy``
ile - basitlestirilmis ama dogru calisan bir alt kumesidir.

Alt moduller
------------
optimize
    Ortalama-varyans optimizasyonu (``mean_variance_optimize``,
    ``efficient_risk``) ve Ledoit-Wolf kovaryans shrinkage
    (``covariance_shrinkage``).
allocation
    Hierarchical Risk Parity (``hrp_optimize``) ve surekli agirliklarin
    tam-sayi hisse/lot adedine cevrilmesi (``discrete_allocation``).
compare
    Fiyat serilerini ortak bir baslangic degerine indirgeme (``rebase``)
    ve coklu strateji ozet performans tablosu (``compare_strategies``).
"""

from __future__ import annotations

from quanttro.portfolio.allocation import discrete_allocation, hrp_optimize
from quanttro.portfolio.compare import compare_strategies, rebase
from quanttro.portfolio.optimize import (
    covariance_shrinkage,
    efficient_risk,
    mean_variance_optimize,
)

__all__ = [
    # optimize
    "mean_variance_optimize",
    "efficient_risk",
    "covariance_shrinkage",
    # allocation
    "hrp_optimize",
    "discrete_allocation",
    # compare
    "rebase",
    "compare_strategies",
]
