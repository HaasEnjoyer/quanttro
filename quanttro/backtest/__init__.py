"""quanttro.backtest: walk-forward bolme, model fabrikasi, risk/performans
metrikleri ve kesitsel siralama (IC) degerlendirme araclari.

train_model.py icindeki dagınık mantigin (walk_forward_folds, make_model,
run_ranking) genellestirilmis/yeniden kullanilabilir hali.
"""
from quanttro.backtest.metrics import (
    alpha_beta,
    avg_loss,
    avg_win,
    calmar_ratio,
    capture_ratio,
    common_sense_ratio,
    conditional_value_at_risk,
    deflated_sharpe_ratio,
    gain_to_pain_ratio,
    hit_rate,
    kelly_criterion,
    max_drawdown,
    omega_ratio,
    payoff_ratio,
    profit_factor,
    recovery_factor,
    risk_of_ruin,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    ulcer_index,
    value_at_risk,
)
from quanttro.backtest.models import ModelFactory, make_model
from quanttro.backtest.ranking import (
    information_coefficient,
    rolling_ic,
    tertile_spread,
)
from quanttro.backtest.walkforward import Fold, WalkForwardSplitter

__all__ = [
    "Fold",
    "WalkForwardSplitter",
    "ModelFactory",
    "make_model",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "profit_factor",
    "deflated_sharpe_ratio",
    "information_coefficient",
    "tertile_spread",
    "rolling_ic",
    "alpha_beta",
    "omega_ratio",
    "tail_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "capture_ratio",
    "avg_win",
    "avg_loss",
    "payoff_ratio",
    "kelly_criterion",
    "risk_of_ruin",
    "ulcer_index",
    "common_sense_ratio",
    "recovery_factor",
    "gain_to_pain_ratio",
]
