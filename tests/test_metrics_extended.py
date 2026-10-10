"""quanttro.backtest.metrics icindeki GENISLETILMIS (QuantStats/empyrical
benzeri) metrikler icin birim testleri.

test_backtest.py'deki konvansiyonu takip eder: her fonksiyon, bagimsiz
(numpy/pandas ile elde ya da elle) hesaplanan beklenen degerle
karsilastirilir. Internet erisimi GEREKTIRMEZ. pytest ile calistirilir::

    python -m pytest tests/test_metrics_extended.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttro.backtest.metrics import (
    alpha_beta,
    avg_loss,
    avg_win,
    capture_ratio,
    common_sense_ratio,
    conditional_value_at_risk,
    gain_to_pain_ratio,
    kelly_criterion,
    omega_ratio,
    payoff_ratio,
    recovery_factor,
    risk_of_ruin,
    tail_ratio,
    ulcer_index,
    value_at_risk,
)

# --------------------------------------------------------------------------
# alpha_beta
# --------------------------------------------------------------------------


def test_alpha_beta_exact_linear_relationship():
    # r = 0.5 * b + 0.001 (sabit alfa, gurultu yok) -> beta tam 0.5,
    # yillliklandirilmamis (periyot bazli) alfa tam 0.001 olmali.
    benchmark = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.02])
    returns = 0.5 * benchmark + 0.001
    alpha, beta = alpha_beta(returns, benchmark, risk_free=0.0, periods_per_year=252)
    assert beta == pytest.approx(0.5)
    assert alpha == pytest.approx(0.001 * 252)


def test_alpha_beta_insufficient_data_is_nan():
    alpha, beta = alpha_beta(pd.Series([0.01]), pd.Series([0.02]))
    assert np.isnan(alpha)
    assert np.isnan(beta)


def test_alpha_beta_zero_benchmark_variance_is_nan():
    benchmark = pd.Series([0.01] * 10)
    returns = pd.Series(np.linspace(-0.02, 0.02, 10))
    alpha, beta = alpha_beta(returns, benchmark)
    assert np.isnan(alpha)
    assert np.isnan(beta)


# --------------------------------------------------------------------------
# omega_ratio
# --------------------------------------------------------------------------


def test_omega_ratio_matches_manual_calculation():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    # gains = 0.02+0.03+0.01=0.06, losses = |-0.01-0.02| = 0.03
    assert omega_ratio(returns, required_return=0.0) == pytest.approx(2.0)


def test_omega_ratio_no_losses_is_inf():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert omega_ratio(returns) == float("inf")


# --------------------------------------------------------------------------
# tail_ratio
# --------------------------------------------------------------------------


def test_tail_ratio_symmetric_distribution_is_one():
    # -50..50 (101 deger): np.percentile linear interpolasyonla
    # 95. persentil = 45, 5. persentil = -45 -> oran tam 1.0
    returns = pd.Series(range(-50, 51))
    assert tail_ratio(returns) == pytest.approx(1.0)


def test_tail_ratio_matches_percentile_definition():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0, 0.02, 500))
    expected = np.percentile(returns, 95) / abs(np.percentile(returns, 5))
    assert tail_ratio(returns) == pytest.approx(expected)


# --------------------------------------------------------------------------
# value_at_risk / conditional_value_at_risk
# --------------------------------------------------------------------------


def test_value_at_risk_matches_known_percentile():
    returns = pd.Series(range(-50, 51))  # -50..50
    # 5. persentil index = 0.05 * (101-1) = 5 -> deger -45
    assert value_at_risk(returns, cutoff=0.05) == pytest.approx(-45.0)


def test_value_at_risk_invalid_cutoff_raises():
    with pytest.raises(ValueError):
        value_at_risk(pd.Series([0.01, 0.02]), cutoff=1.5)


def test_conditional_value_at_risk_is_mean_of_tail_beyond_var():
    returns = pd.Series(range(-50, 51))  # -50..50
    # VaR(0.05) = -45; kuyruk = {-50,...,-45} (6 deger), ortalama = -47.5
    assert conditional_value_at_risk(returns, cutoff=0.05) == pytest.approx(-47.5)


def test_cvar_magnitude_at_least_var_magnitude():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0, 0.02, 300))
    var = value_at_risk(returns, 0.05)
    cvar = conditional_value_at_risk(returns, 0.05)
    assert abs(cvar) >= abs(var)


# --------------------------------------------------------------------------
# capture_ratio
# --------------------------------------------------------------------------


def test_capture_ratio_up_and_down():
    benchmark = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    returns = 2.0 * benchmark  # strateji her zaman benchmark'in 2 kati
    assert capture_ratio(returns, benchmark, side="up") == pytest.approx(2.0)
    assert capture_ratio(returns, benchmark, side="down") == pytest.approx(2.0)


def test_capture_ratio_invalid_side_raises():
    with pytest.raises(ValueError):
        capture_ratio(pd.Series([0.01]), pd.Series([0.01]), side="sideways")


# --------------------------------------------------------------------------
# avg_win / avg_loss / payoff_ratio
# --------------------------------------------------------------------------

_SAMPLE_RETURNS = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])


def test_avg_win_and_avg_loss_manual_calculation():
    assert avg_win(_SAMPLE_RETURNS) == pytest.approx((0.02 + 0.03 + 0.01) / 3)
    assert avg_loss(_SAMPLE_RETURNS) == pytest.approx((-0.01 - 0.02) / 2)


def test_payoff_ratio_manual_calculation():
    # avg_win = 0.02, avg_loss = -0.015 -> payoff = 0.02/0.015 = 4/3
    assert payoff_ratio(_SAMPLE_RETURNS) == pytest.approx(4.0 / 3.0)


def test_avg_win_no_wins_is_nan():
    assert np.isnan(avg_win(pd.Series([-0.01, -0.02])))


def test_avg_loss_no_losses_is_nan():
    assert np.isnan(avg_loss(pd.Series([0.01, 0.02])))


# --------------------------------------------------------------------------
# kelly_criterion
# --------------------------------------------------------------------------


def test_kelly_criterion_manual_calculation():
    # win_rate = 3/5 = 0.6, win_loss_ratio = 4/3
    # f* = 0.6 - 0.4/(4/3) = 0.6 - 0.3 = 0.3
    assert kelly_criterion(_SAMPLE_RETURNS) == pytest.approx(0.3)


def test_kelly_criterion_no_losses_is_nan():
    assert np.isnan(kelly_criterion(pd.Series([0.01, 0.02, 0.03])))


# --------------------------------------------------------------------------
# risk_of_ruin
# --------------------------------------------------------------------------


def test_risk_of_ruin_manual_calculation():
    # win_rate = 0.6 -> edge = 0.2, risk_fraction=1.0 -> units=1
    # RoR = (0.8/1.2)^1 = 2/3
    assert risk_of_ruin(_SAMPLE_RETURNS, risk_fraction=1.0) == pytest.approx(2.0 / 3.0)


def test_risk_of_ruin_edge_le_zero_is_certain_ruin():
    returns = pd.Series([0.01, -0.01, -0.02, 0.005, -0.03])  # win_rate=0.4 <= 0.5
    assert risk_of_ruin(returns) == pytest.approx(1.0)


def test_risk_of_ruin_no_losses_is_zero():
    returns = pd.Series([0.01, 0.02, 0.03])  # win_rate=1.0 -> edge=1.0
    assert risk_of_ruin(returns) == pytest.approx(0.0)


def test_risk_of_ruin_smaller_risk_fraction_lowers_ruin_probability():
    big_risk = risk_of_ruin(_SAMPLE_RETURNS, risk_fraction=1.0)
    small_risk = risk_of_ruin(_SAMPLE_RETURNS, risk_fraction=0.1)
    assert small_risk < big_risk


# --------------------------------------------------------------------------
# ulcer_index
# --------------------------------------------------------------------------


def test_ulcer_index_matches_manual_calculation():
    curve = pd.Series([1.0, 1.1, 1.0, 0.9, 0.95, 1.0])
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    expected = np.sqrt(np.mean(drawdown.to_numpy() ** 2))
    assert ulcer_index(curve) == pytest.approx(expected)


def test_ulcer_index_no_drawdown_is_zero():
    curve = pd.Series([1.0, 1.01, 1.02, 1.03])  # monotonik artan, drawdown yok
    assert ulcer_index(curve) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# common_sense_ratio
# --------------------------------------------------------------------------


def test_common_sense_ratio_symmetric_distribution():
    # -50..50: tail_ratio = 1.0, profit_factor = 1275/1275 = 1.0 -> CSR = 1.0
    returns = pd.Series(range(-50, 51))
    assert common_sense_ratio(returns) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# recovery_factor
# --------------------------------------------------------------------------


def test_recovery_factor_manual_calculation():
    curve = pd.Series([1.0, 1.2, 1.1, 0.8, 1.5])
    # toplam_getiri = 1.5/1.0 - 1 = 0.5
    # max_drawdown: running_max=[1,1.2,1.2,1.2,1.5]; en kotu dusus 0.8/1.2-1=-1/3
    assert recovery_factor(curve) == pytest.approx(0.5 / (1.0 / 3.0))


def test_recovery_factor_insufficient_data_is_nan():
    assert np.isnan(recovery_factor(pd.Series([1.0])))


# --------------------------------------------------------------------------
# gain_to_pain_ratio
# --------------------------------------------------------------------------


def test_gain_to_pain_ratio_manual_calculation():
    # toplam_getiri = 0.02-0.01+0.03-0.02+0.01 = 0.03
    # toplam_kayip = |-0.01-0.02| = 0.03 -> oran = 1.0
    assert gain_to_pain_ratio(_SAMPLE_RETURNS) == pytest.approx(1.0)


def test_gain_to_pain_ratio_no_losses_is_inf():
    assert gain_to_pain_ratio(pd.Series([0.01, 0.02])) == float("inf")
