"""
quanttro.portfolio icin pytest testleri.

Her fonksiyon icin SENTETIK (internet gerektirmeyen) veri kullanilir;
mumkun oldugunda sonuclar bagimsiz/elle hesaplanan (kapali-form) degerlerle
karsilastirilir (orn. iki-varlikli minimum-varyans portfoyu icin klasik
kapali-form formul).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttro.portfolio import allocation, compare
from quanttro.portfolio import optimize as opt


# ---------------------------------------------------------------------------
# optimize.py
# ---------------------------------------------------------------------------

class TestMeanVarianceOptimize:
    def _two_asset_setup(self):
        # A ve B negatif korelasyonlu (corr=-0.5), farkli varyansli iki varlik.
        var_a, var_b, corr = 0.04, 0.09, -0.5
        cov_ab = corr * np.sqrt(var_a) * np.sqrt(var_b)
        cov_matrix = pd.DataFrame(
            {"A": [var_a, cov_ab], "B": [cov_ab, var_b]}, index=["A", "B"]
        )
        expected_returns = pd.Series({"A": 0.08, "B": 0.10})
        return expected_returns, cov_matrix, var_a, var_b, cov_ab

    def test_min_volatility_matches_closed_form_two_asset(self):
        expected_returns, cov_matrix, var_a, var_b, cov_ab = self._two_asset_setup()
        weights = opt.mean_variance_optimize(expected_returns, cov_matrix, objective="min_volatility")

        # iki-varlikli minimum-varyans portfoyu icin kapali-form formul
        w_a_expected = (var_b - cov_ab) / (var_a + var_b - 2 * cov_ab)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights["A"] == pytest.approx(w_a_expected, abs=1e-3)
        assert weights["B"] == pytest.approx(1 - w_a_expected, abs=1e-3)

    def test_negative_correlation_leads_to_sensible_diversification(self):
        # negatif korelasyon diversifikasyonu TESVIK eder: ne min-vol ne de
        # max-sharpe cozumu tek bir varlikta (corner solution) toplanmamali.
        expected_returns, cov_matrix, *_ = self._two_asset_setup()

        for objective in ("min_volatility", "max_sharpe"):
            weights = opt.mean_variance_optimize(expected_returns, cov_matrix, objective=objective)
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)
            assert (weights > 0.05).all(), f"{objective} bir varlikta toplanmis (cesitlendirme yok)"
            assert (weights < 0.95).all(), f"{objective} bir varlikta toplanmis (cesitlendirme yok)"

    def test_max_sharpe_beats_either_pure_asset_portfolio(self):
        expected_returns, cov_matrix, *_ = self._two_asset_setup()
        weights = opt.mean_variance_optimize(expected_returns, cov_matrix, objective="max_sharpe")

        mu = expected_returns.to_numpy()
        cov = cov_matrix.to_numpy()

        def sharpe(w: np.ndarray) -> float:
            return (w @ mu) / np.sqrt(w @ cov @ w)

        sharpe_opt = sharpe(weights.to_numpy())
        sharpe_pure_a = sharpe(np.array([1.0, 0.0]))
        sharpe_pure_b = sharpe(np.array([0.0, 1.0]))

        assert sharpe_opt >= max(sharpe_pure_a, sharpe_pure_b) - 1e-9

    def test_invalid_objective_raises(self):
        expected_returns, cov_matrix, *_ = self._two_asset_setup()
        with pytest.raises(ValueError):
            opt.mean_variance_optimize(expected_returns, cov_matrix, objective="not_a_real_objective")


class TestEfficientRisk:
    def test_respects_target_volatility_and_sums_to_one(self):
        expected_returns = pd.Series({"A": 0.08, "B": 0.10, "C": 0.05})
        cov_matrix = pd.DataFrame(
            {
                "A": [0.04, -0.01, 0.00],
                "B": [-0.01, 0.09, 0.01],
                "C": [0.00, 0.01, 0.02],
            },
            index=["A", "B", "C"],
        )
        target_vol = 0.18
        weights = opt.efficient_risk(expected_returns, cov_matrix, target_volatility=target_vol)

        achieved_vol = float(np.sqrt(weights.to_numpy() @ cov_matrix.to_numpy() @ weights.to_numpy()))
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert achieved_vol <= target_vol + 1e-4

    def test_unreachable_target_raises(self):
        expected_returns = pd.Series({"A": 0.08, "B": 0.10})
        cov_matrix = pd.DataFrame({"A": [0.04, 0.0], "B": [0.0, 0.09]}, index=["A", "B"])
        with pytest.raises(ValueError):
            # min mumkun vol (iki varligin negatif olmayan karisimlariyla) 0.2'den buyuk
            opt.efficient_risk(expected_returns, cov_matrix, target_volatility=0.001)


class TestCovarianceShrinkage:
    def test_ledoit_wolf_is_symmetric_positive_semidefinite(self):
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.normal(0, 0.02, size=(100, 4)), columns=list("ABCD"))
        cov = opt.covariance_shrinkage(returns, method="ledoit_wolf")

        np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
        assert (eigenvalues >= -1e-10).all()

    def test_sample_method_matches_pandas_cov(self):
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.normal(0, 0.02, size=(50, 3)), columns=list("XYZ"))
        cov = opt.covariance_shrinkage(returns, method="sample")
        pd.testing.assert_frame_equal(cov, returns.cov())


# ---------------------------------------------------------------------------
# allocation.py
# ---------------------------------------------------------------------------

class TestHRPOptimize:
    def _synthetic_returns(self, n_assets=5, n_obs=200, seed=42):
        rng = np.random.default_rng(seed)
        factor = rng.normal(0, 0.01, size=n_obs)
        data = {}
        for i in range(n_assets):
            idiosyncratic = rng.normal(0, 0.01, size=n_obs)
            data[f"S{i}"] = 0.5 * factor + idiosyncratic
        return pd.DataFrame(data)

    def test_weights_sum_to_one_and_non_negative(self):
        returns = self._synthetic_returns()
        weights = allocation.hrp_optimize(returns)

        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert (weights >= 0).all()
        assert set(weights.index) == set(returns.columns)

    def test_single_asset_raises(self):
        returns = pd.DataFrame({"A": np.random.default_rng(2).normal(0, 0.01, 50)})
        with pytest.raises(ValueError):
            allocation.hrp_optimize(returns)


class TestDiscreteAllocation:
    def test_total_spend_does_not_exceed_total_value(self):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        prices = pd.Series({"A": 37.5, "B": 112.0, "C": 8.25})
        total_value = 10_000.0

        result = allocation.discrete_allocation(weights, prices, total_value)
        spend = sum(count * prices[sym] for sym, count in result["allocation"].items())

        assert spend <= total_value + 1e-9
        assert result["leftover_cash"] == pytest.approx(total_value - spend, abs=1e-6)
        assert result["leftover_cash"] >= -1e-9

    def test_all_allocations_are_nonnegative_integers(self):
        weights = pd.Series({"A": 0.6, "B": 0.4})
        prices = pd.Series({"A": 999.0, "B": 5.0})
        result = allocation.discrete_allocation(weights, prices, total_value=1000.0)
        for count in result["allocation"].values():
            assert isinstance(count, int)
            assert count >= 0

    def test_unnormalized_weights_are_normalized(self):
        # toplami 1 olmayan agirliklar (orn. 2.0) once normalize edilmeli
        weights = pd.Series({"A": 1.0, "B": 1.0})
        prices = pd.Series({"A": 50.0, "B": 50.0})
        result = allocation.discrete_allocation(weights, prices, total_value=1000.0)
        spend = sum(count * prices[sym] for sym, count in result["allocation"].items())
        assert spend <= 1000.0 + 1e-9


# ---------------------------------------------------------------------------
# compare.py
# ---------------------------------------------------------------------------

class TestRebase:
    def test_first_row_equals_base_value(self):
        prices = pd.DataFrame(
            {"A": [120.0, 130.0, 110.0], "B": [5.0, 5.5, 6.0]}
        )
        out = compare.rebase(prices, base_value=100.0)
        np.testing.assert_allclose(out.iloc[0].to_numpy(), [100.0, 100.0])

    def test_custom_base_value_and_proportions_preserved(self):
        prices = pd.DataFrame({"A": [10.0, 20.0, 5.0]})
        out = compare.rebase(prices, base_value=50.0)
        assert out["A"].iloc[0] == pytest.approx(50.0)
        # A ikiye katlaniyor sonra ceyreklenmis -> rebase edilmis deger de aynen oyle davranmali
        assert out["A"].iloc[1] == pytest.approx(100.0)
        assert out["A"].iloc[2] == pytest.approx(25.0)


class TestCompareStrategies:
    def test_output_shape_and_columns(self):
        rng = np.random.default_rng(3)
        returns_dict = {
            "strategy_1": pd.Series(rng.normal(0.001, 0.01, 100)),
            "strategy_2": pd.Series(rng.normal(0.0005, 0.02, 100)),
        }
        out = compare.compare_strategies(returns_dict)

        assert list(out.index) == ["strategy_1", "strategy_2"]
        assert list(out.columns) == [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]

    def test_matches_metrics_module_for_sharpe_and_drawdown(self):
        from quanttro.backtest.metrics import max_drawdown as mdd_fn
        from quanttro.backtest.metrics import sharpe_ratio as sharpe_fn

        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.015, -0.005])
        out = compare.compare_strategies({"s": returns})

        cumulative = (1 + returns).cumprod()
        assert out.loc["s", "sharpe_ratio"] == pytest.approx(sharpe_fn(returns))
        assert out.loc["s", "max_drawdown"] == pytest.approx(mdd_fn(cumulative))

    def test_short_series_yields_nan(self):
        out = compare.compare_strategies({"s": pd.Series([0.01])})
        assert out.loc["s"].isna().all()
