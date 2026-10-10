"""quanttro.robustness: backtest saglamlik (robustness) testleri.

Ana akim acik kaynak backtest kutuphanelerinde (vectorbt, backtrader,
zipline) duzgun/bakimli halde bulunmayan, akademik olarak tanimli
saglamlik araclarini toplar:

- :mod:`quanttro.robustness.cpcv` - Combinatorial Purged Cross-Validation
  (Lopez de Prado, 2018).
- :mod:`quanttro.robustness.overfitting` - Probability of Backtest
  Overfitting / CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2017).
- :mod:`quanttro.robustness.montecarlo` - getiri serisi Monte Carlo
  yeniden-orneklemesi (block bootstrap / shuffle / iid bootstrap).
- :mod:`quanttro.robustness.sensitivity` - tek-tek (one-at-a-time)
  parametre hassasiyeti analizi ve plato/ucurum siniflandirmasi.
"""
from quanttro.robustness.cpcv import CombinatorialPurgedCV
from quanttro.robustness.montecarlo import MonteCarloSimulator
from quanttro.robustness.overfitting import probability_of_backtest_overfitting
from quanttro.robustness.sensitivity import classify_sensitivity, parameter_sensitivity

__all__ = [
    "CombinatorialPurgedCV",
    "probability_of_backtest_overfitting",
    "MonteCarloSimulator",
    "parameter_sensitivity",
    "classify_sensitivity",
]
