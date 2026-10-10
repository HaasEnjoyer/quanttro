"""quanttro.backtest paketi icin birim testleri.

Her fonksiyon, sentetik veriler uzerinde - formulun bagimsiz (numpy ile
elde) hesaplanan degeriyle karsilastirilarak ya da bilinen sinir
durumlariyla - dogrulanir. pytest ile calistirilir::

    python -m pytest tests/test_backtest.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quanttro.backtest.metrics import (
    calmar_ratio,
    deflated_sharpe_ratio,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)
from quanttro.backtest.models import ModelFactory, make_model
from quanttro.backtest.ranking import (
    information_coefficient,
    rolling_ic,
    tertile_spread,
)
from quanttro.backtest.walkforward import WalkForwardSplitter

# --------------------------------------------------------------------------
# metrics.py
# --------------------------------------------------------------------------


def test_sharpe_ratio_matches_manual_calculation():
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.008, -0.012, 0.02])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns, periods_per_year=252) == pytest.approx(expected)


def test_sharpe_ratio_applies_annual_risk_free_rate_correctly():
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.008])
    rf_annual = 0.05
    excess = returns - rf_annual / 252
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns, periods_per_year=252, risk_free=rf_annual) == pytest.approx(expected)


def test_sharpe_ratio_constant_returns_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 10)))


def test_sortino_ratio_matches_manual_calculation():
    returns = pd.Series([0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.015])
    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    expected = returns.mean() / downside_dev * np.sqrt(252)
    assert sortino_ratio(returns, periods_per_year=252) == pytest.approx(expected)


def test_sortino_ratio_ignores_upside_volatility_unlike_sharpe():
    # Sharpe toplam (iki yonlu) oynakligi cezalandirir, Sortino sadece
    # asagi yonlu sapmayi cezalandirir; asimetrik (sol carpik) bir seride
    # bu iki oran birbirinden farkli olmalidir.
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, -0.10])
    assert sharpe_ratio(returns) != pytest.approx(sortino_ratio(returns))


def test_max_drawdown_known_curve():
    curve = pd.Series([1.0, 1.1, 1.05, 0.9, 1.2, 0.8])
    # el ile: running_max = [1.0,1.1,1.1,1.1,1.2,1.2]
    # drawdown en kotu noktada 0.8/1.2 - 1 = -1/3
    assert max_drawdown(curve) == pytest.approx(-1.0 / 3.0)


def test_max_drawdown_monotonic_increasing_curve_is_zero():
    curve = pd.Series([1.0, 1.05, 1.1, 1.2, 1.3])
    assert max_drawdown(curve) == pytest.approx(0.0)


def test_calmar_ratio_matches_manual_calculation():
    returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.02, 0.01, -0.015, 0.025])
    n = len(returns)
    cumulative = (1 + returns).cumprod()
    mdd = max_drawdown(cumulative)
    annualized_return = cumulative.iloc[-1] ** (252 / n) - 1
    expected = annualized_return / abs(mdd)
    assert calmar_ratio(returns, periods_per_year=252) == pytest.approx(expected)


def test_hit_rate_counts_strictly_positive_periods():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0])
    assert hit_rate(returns) == pytest.approx(2 / 5)


def test_profit_factor_matches_manual_calculation():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.0])
    gains = 0.01 + 0.03
    losses = 0.02 + 0.01
    assert profit_factor(returns) == pytest.approx(gains / losses)


def test_profit_factor_no_losses_is_infinite():
    assert profit_factor(pd.Series([0.01, 0.02, 0.0])) == float("inf")


def test_deflated_sharpe_ratio_single_trial_matches_probabilistic_sharpe_ratio():
    # n_trials=1 oldugunda DSR, 0 karsiliginda klasik Probabilistic
    # Sharpe Ratio'ya (PSR) esit olmalidir (coklu deneme duzeltmesi yok).
    sharpe, skew, kurt, n_obs = 1.2, 0.0, 3.0, 252
    dsr = deflated_sharpe_ratio(sharpe, n_trials=1, skewness=skew, kurtosis=kurt, n_observations=n_obs)

    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_obs - 1))
    expected = norm.cdf(sharpe / se)
    assert dsr == pytest.approx(expected)


def test_deflated_sharpe_ratio_decreases_with_more_trials():
    # Coklu test duzeltmesi daha fazla deneme sayisinda daha katı olmali:
    # ayni gozlemlenen Sharpe icin DSR dusmeli. (Not: periyot bazli Sharpe
    # gercekci bir degerde (0.1) tutulur; cok yuksek bir deger norm.cdf'yi
    # tum senaryolarda 1.0'a saturate ederek testi anlamsizlastirir.)
    kwargs = dict(sharpe=0.1, skewness=0.0, kurtosis=3.0, n_observations=252)
    dsr_1 = deflated_sharpe_ratio(n_trials=1, **kwargs)
    dsr_10 = deflated_sharpe_ratio(n_trials=10, **kwargs)
    dsr_100 = deflated_sharpe_ratio(n_trials=100, **kwargs)
    assert dsr_1 > dsr_10 > dsr_100


def test_deflated_sharpe_ratio_is_a_valid_probability():
    dsr = deflated_sharpe_ratio(sharpe=0.8, n_trials=20, skewness=-0.3, kurtosis=4.5, n_observations=500)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_rejects_invalid_n_trials():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sharpe=1.0, n_trials=0, skewness=0.0, kurtosis=3.0, n_observations=100)


# --------------------------------------------------------------------------
# ranking.py
# --------------------------------------------------------------------------


def test_information_coefficient_perfect_positive_correlation():
    idx = pd.date_range("2024-01-01", periods=20)
    predicted = pd.Series(np.arange(20), index=idx)
    actual = pd.Series(np.arange(20) * 2.0 + 1.0, index=idx)
    assert information_coefficient(predicted, actual) == pytest.approx(1.0)


def test_information_coefficient_perfect_negative_correlation():
    idx = pd.date_range("2024-01-01", periods=20)
    predicted = pd.Series(np.arange(20), index=idx)
    actual = pd.Series(-np.arange(20), index=idx)
    assert information_coefficient(predicted, actual) == pytest.approx(-1.0)


def test_information_coefficient_returns_nan_with_too_few_points():
    predicted = pd.Series([1.0, 2.0])
    actual = pd.Series([2.0, 1.0])
    assert np.isnan(information_coefficient(predicted, actual))


def test_tertile_spread_separates_top_and_bottom_groups():
    # 9 hisse: tahmin sirasina gore gerceklesen getiri de monotonik artiyor.
    predicted = pd.Series(np.arange(9))
    actual = pd.Series(np.arange(9) * 0.01)
    expected = actual.iloc[-3:].mean() - actual.iloc[:3].mean()
    assert tertile_spread(predicted, actual, n_groups=3) == pytest.approx(expected)


def test_tertile_spread_supports_quintiles():
    predicted = pd.Series(np.arange(10))
    actual = pd.Series(np.arange(10) * 0.02)
    expected = actual.iloc[-2:].mean() - actual.iloc[:2].mean()  # grup buyuklugu = 10 // 5 = 2
    assert tertile_spread(predicted, actual, n_groups=5) == pytest.approx(expected)


def test_tertile_spread_rejects_too_few_groups():
    with pytest.raises(ValueError):
        tertile_spread(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), n_groups=1)


def test_rolling_ic_perfect_prediction_gives_ic_of_one():
    dates = pd.date_range("2024-01-01", periods=100)
    symbols = [f"SYM{i}" for i in range(10)]
    rng = np.random.default_rng(42)
    actual_by_date = pd.DataFrame(rng.normal(size=(100, 10)), index=dates, columns=symbols)
    predicted_by_date = actual_by_date.copy()  # mukemmel tahmin -> her gun IC = 1

    result = rolling_ic(predicted_by_date, actual_by_date, window=20)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid > 0.999).all()


def test_rolling_ic_returns_series_indexed_like_input_dates():
    dates = pd.date_range("2024-01-01", periods=30)
    symbols = [f"SYM{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    actual = pd.DataFrame(rng.normal(size=(30, 6)), index=dates, columns=symbols)
    predicted = pd.DataFrame(rng.normal(size=(30, 6)), index=dates, columns=symbols)

    result = rolling_ic(predicted, actual, window=10)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(dates)


# --------------------------------------------------------------------------
# walkforward.py
# --------------------------------------------------------------------------


def _business_days(n_days: int = 100) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02", periods=n_days, freq="B")


def test_expanding_window_produces_requested_fold_count_and_order():
    index = _business_days(100)
    folds = list(WalkForwardSplitter.expanding_window(n_folds=4).split(index))
    assert len(folds) == 4
    for train_cutoff, test_start, test_end in folds:  # 3-tuple unpack (geriye uyumlu)
        assert train_cutoff < test_start <= test_end


def test_expanding_window_train_size_never_shrinks():
    index = _business_days(100)
    folds = list(WalkForwardSplitter.expanding_window(n_folds=4).split(index))
    train_sizes = [(index <= fold.train_cutoff).sum() for fold in folds]
    assert all(a <= b for a, b in zip(train_sizes, train_sizes[1:]))


def test_expanding_window_folds_are_chronological_and_non_overlapping():
    index = _business_days(120)
    folds = list(WalkForwardSplitter.expanding_window(n_folds=5).split(index))
    for (_, _, prev_end), (_, next_start, _) in zip(folds, folds[1:]):
        assert prev_end < next_start


def test_purge_gap_removes_dates_near_train_test_boundary():
    index = _business_days(100)
    unique_dates = index.unique().sort_values()

    folds_no_purge = list(WalkForwardSplitter(n_folds=4, purge_gap=0).split(index))
    folds_purged = list(WalkForwardSplitter(n_folds=4, purge_gap=5).split(index))

    for fold_a, fold_b in zip(folds_no_purge, folds_purged):
        assert fold_a.test_start == fold_b.test_start  # test blogu purge'den etkilenmez
        assert fold_b.train_cutoff < fold_a.train_cutoff  # train, purge sayesinde daha erken kesiliyor
        gap_dates = unique_dates[(unique_dates > fold_b.train_cutoff) & (unique_dates < fold_b.test_start)]
        assert len(gap_dates) == 5


def test_rolling_window_keeps_train_width_approximately_fixed():
    index = _business_days(120)
    n_folds = 5
    folds = list(WalkForwardSplitter.rolling_window(n_folds=n_folds, train_window=1).split(index))
    assert len(folds) == n_folds

    train_lengths = [
        ((index >= fold.train_start) & (index <= fold.train_cutoff)).sum() for fold in folds
    ]
    # Sabit genislikte kayan pencere: tum fold'larda genislik birbirine
    # (array_split'in esit-olmayan bolunme payi haric) cok yakin olmali.
    assert max(train_lengths) - min(train_lengths) <= 1


def test_rolling_window_train_set_is_narrower_than_expanding_in_later_folds():
    index = _business_days(150)
    n_folds = 6
    exp_folds = list(WalkForwardSplitter.expanding_window(n_folds=n_folds).split(index))
    roll_folds = list(WalkForwardSplitter.rolling_window(n_folds=n_folds, train_window=2).split(index))

    exp_last, roll_last = exp_folds[-1], roll_folds[-1]
    exp_train_len = (index <= exp_last.train_cutoff).sum()
    roll_train_len = ((index >= roll_last.train_start) & (index <= roll_last.train_cutoff)).sum()

    assert roll_train_len < exp_train_len


def test_split_raises_when_not_enough_unique_dates():
    index = _business_days(3)
    with pytest.raises(ValueError):
        list(WalkForwardSplitter(n_folds=5).split(index))


def test_constructor_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        WalkForwardSplitter(n_folds=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(purge_gap=-1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(window="sideways")


# --------------------------------------------------------------------------
# models.py
# --------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"] * 2.0 - X["f1"] + rng.normal(scale=0.1, size=300)
    return X, y


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] + rng.normal(scale=0.1, size=300) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("model_type", ModelFactory.available_models())
def test_model_factory_regression_fit_predict(model_type, regression_data):
    X, y = regression_data
    model = ModelFactory.create(model_type, task="regression")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("model_type", ModelFactory.available_models())
def test_model_factory_classification_fit_predict(model_type, classification_data):
    X, y = classification_data
    model = ModelFactory.create(model_type, task="classification")
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_model_factory_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        ModelFactory.create("not_a_model", task="regression")


def test_model_factory_rejects_unknown_task():
    with pytest.raises(ValueError):
        ModelFactory.create("lightgbm", task="not_a_task")


def test_model_factory_applies_hyperparameter_overrides():
    model = ModelFactory.create("lightgbm", task="regression", n_estimators=10)
    assert model.get_params()["n_estimators"] == 10


def test_make_model_is_backward_compatible_and_defaults_to_lightgbm():
    model = make_model("regression")
    assert type(model).__name__ == "LGBMRegressor"
