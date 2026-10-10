"""quanttro.robustness paketi icin birim testleri.

CPCV ve Monte Carlo testleri, cogunlukla ELLE dogrulanabilir sentetik
pozisyon/getiri dizileri uzerinde kurulur. PBO testleri ise bu modulun
TEMEL degerini (overfitting tespiti) dogrudan kanitlar: tamamen
rastgele (sinyalsiz) denemelerde PBO'nun yuksek, tek bir tutarli/gercek
sinyali olan senaryoda ise DUSUK cikmasi beklenir - ikisi de burada
test edilir. pytest ile calistirilir::

    python -m pytest tests/test_robustness.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quanttro.robustness.cpcv import CombinatorialPurgedCV
from quanttro.robustness.montecarlo import MonteCarloSimulator
from quanttro.robustness.overfitting import probability_of_backtest_overfitting
from quanttro.robustness.sensitivity import classify_sensitivity, parameter_sensitivity

# --------------------------------------------------------------------------
# cpcv.py
# --------------------------------------------------------------------------


def test_cpcv_n_splits_matches_binomial_coefficient():
    assert CombinatorialPurgedCV(n_groups=6, n_test_groups=2).n_splits() == math.comb(6, 2)
    assert CombinatorialPurgedCV(n_groups=10, n_test_groups=3).n_splits() == math.comb(10, 3)
    assert CombinatorialPurgedCV(n_groups=5, n_test_groups=1).n_splits() == math.comb(5, 1)


def test_cpcv_split_yields_exactly_n_splits_combinations():
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    splits = list(cv.split(index))
    assert len(splits) == cv.n_splits()


def test_cpcv_train_and_test_are_disjoint_in_every_combination():
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    for train_idx, test_idx in cv.split(index):
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        assert len(test_idx) > 0
        assert len(train_idx) > 0


def test_cpcv_no_purge_no_embargo_train_and_test_cover_everything():
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=0, embargo_pct=0.0)
    train_idx, test_idx = next(cv.split(index))
    assert set(train_idx.tolist()) | set(test_idx.tolist()) == set(range(60))


def test_cpcv_purge_gap_removes_boundary_adjacent_train_positions():
    # n=60, 6 esit grup (10'ar pozisyon): grup0=[0..9], grup1=[10..19], ...
    # Kombinasyonlar leksikografik uretilir, ilki test gruplari=(0, 1).
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=3)
    train_idx, test_idx = next(cv.split(index))

    assert list(test_idx) == list(range(20))
    # purge: grup1 (test) sonrasindaki ilk 3 pozisyon (20,21,22) egitimden
    # cikarilmali -> train, 23'ten baslamali.
    assert train_idx.min() == 23
    assert set([20, 21, 22]).isdisjoint(train_idx.tolist())


def test_cpcv_purge_gap_applies_at_each_scattered_test_group_boundary():
    # Test gruplari ardisik olmak zorunda degil: ucuncu kombinasyon (0, 3)
    # -> grup0=[0..9] (test), grup3=[30..39] (test), aralarinda train var.
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=3)
    splits = list(cv.split(index))
    train_idx, test_idx = splits[2]

    assert list(test_idx) == list(range(0, 10)) + list(range(30, 40))
    purged_positions = {10, 11, 12, 27, 28, 29, 40, 41, 42}
    assert purged_positions.isdisjoint(set(train_idx.tolist()))
    assert set(train_idx.tolist()) == set(range(13, 27)) | set(range(43, 60))


def test_cpcv_embargo_extends_gap_after_test_group():
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_gap=0, embargo_pct=0.05)
    train_idx, _ = next(cv.split(index))
    # embargo_size = round(60 * 0.05) = 3 -> grup1 (test) sonrasi 3 pozisyon
    # (20, 21, 22) de egitimden cikarilmali (purge_gap=0 olsa bile).
    assert train_idx.min() == 23


def test_cpcv_rejects_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=5)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=2, purge_gap=-1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_pct=1.0)


# --------------------------------------------------------------------------
# overfitting.py (PBO) - bu modulun TEMEL degerini kanitlayan testler
# --------------------------------------------------------------------------


def test_pbo_is_high_for_purely_random_trials_with_no_real_signal():
    # 40 strateji, 1000 periyot, HICBIRINDE gercek sinyal yok (hepsi
    # ayni (0, 0.01) normal dagilimdan BAGIMSIZ cekilmis gurultu).
    # Sabit tohum (seed=7) ile tam deterministik: IS'te "en iyi" secilen
    # strateji sans eseri secildiginden, OOS'ta medyanin altinda/ustunde
    # kalmasi da sans eseri olmali -> PBO yuksek (medyana yakin/uzeri)
    # cikmalidir.
    rng = np.random.default_rng(7)
    trial_results = pd.DataFrame(rng.normal(loc=0.0, scale=0.01, size=(1000, 40)))

    result = probability_of_backtest_overfitting(trial_results, n_partitions=10)

    assert result["n_combinations_tested"] == math.comb(10, 5)
    assert len(result["logits"]) == result["n_combinations_tested"]
    assert result["pbo"] > 0.4


def test_pbo_is_lower_when_one_strategy_has_a_genuine_consistent_edge():
    # Ayni rastgele matris, ama bir sutun (0) GERCEK ve TUTARLI bir
    # ustunluge sahip (anlamli pozitif drift, tum periyotlarda gecerli) -
    # bu strateji hem IS'te hem OOS'te sistematik olarak iyi performans
    # gostermeli, bu yuzden PBO rastgele senaryoya kiyasla DUSUK cikmali.
    rng = np.random.default_rng(7)
    trial_results = pd.DataFrame(rng.normal(loc=0.0, scale=0.01, size=(1000, 40)))

    signal_rng = np.random.default_rng(777)
    trial_results[0] = signal_rng.normal(loc=0.015, scale=0.01, size=1000)

    random_result = probability_of_backtest_overfitting(
        pd.DataFrame(np.random.default_rng(7).normal(loc=0.0, scale=0.01, size=(1000, 40))),
        n_partitions=10,
    )
    signal_result = probability_of_backtest_overfitting(trial_results, n_partitions=10)

    assert signal_result["pbo"] < random_result["pbo"]
    assert signal_result["pbo"] < 0.2


def test_pbo_rejects_odd_n_partitions():
    trial_results = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 5)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trial_results, n_partitions=9)


def test_pbo_rejects_too_few_trials():
    trial_results = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trial_results, n_partitions=10)


def test_pbo_rejects_nan_values():
    data = np.random.default_rng(0).normal(size=(100, 5))
    data[0, 0] = np.nan
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame(data), n_partitions=10)


def test_pbo_result_is_a_valid_probability():
    trial_results = pd.DataFrame(np.random.default_rng(3).normal(size=(200, 8)))
    result = probability_of_backtest_overfitting(trial_results, n_partitions=8)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_combinations_tested"] == math.comb(8, 4)


# --------------------------------------------------------------------------
# montecarlo.py
# --------------------------------------------------------------------------


def _toy_returns(n: int = 100) -> pd.Series:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(loc=0.001, scale=0.02, size=n), index=idx)


@pytest.mark.parametrize("method", ["block_bootstrap", "shuffle", "iid_bootstrap"])
def test_montecarlo_run_produces_requested_number_of_simulations_of_correct_length(method):
    returns = _toy_returns(100)
    sim = MonteCarloSimulator(method=method, block_size=10, random_state=0)
    curves = sim.run(returns, n_sims=50)

    assert curves.shape == (100, 50)
    assert curves.notna().all().all()
    assert len(curves) == len(returns)


def test_montecarlo_shuffle_preserves_final_cumulative_value():
    # Shuffle sadece SIRAYI degistirir, ayni getiri degerlerinin carpimi
    # (carpma islemi degismeli oldugundan) hangi sirada carpilirsa
    # carpilsin AYNI kalmalidir - bu yuzden her simulasyonun SON degeri,
    # orijinal serinin kumulatif urunune esit olmalidir.
    returns = _toy_returns(80)
    expected_final = float((1.0 + returns).prod())

    sim = MonteCarloSimulator(method="shuffle", random_state=5)
    curves = sim.run(returns, n_sims=30)

    final_values = curves.iloc[-1]
    assert final_values.to_numpy() == pytest.approx(expected_final)


def test_montecarlo_block_bootstrap_sample_blocks_are_contiguous_in_original_series():
    # values[i] = i secilerek, orneklenen degerlerin KENDISI orijinal
    # pozisyonlari temsil eder - bu sayede "ardisik (bitisik) blok"
    # kosulu, basit bir fark (diff) kontroluyle dogrulanabilir.
    n_obs = 50
    values = np.arange(n_obs, dtype=float)
    sim = MonteCarloSimulator(method="block_bootstrap", block_size=10, random_state=42)
    rng = np.random.default_rng(0)

    sample = sim._block_bootstrap_sample(values, n=n_obs, rng=rng)
    assert len(sample) == n_obs

    first_block = sample[:10]
    # ardisik farklar: normalde +1, dairesel (circular) sarma noktasinda
    # -(n_obs - 1) olabilir - modulo n_obs alinca her durumda 1 olmalidir.
    diffs = np.diff(first_block) % n_obs
    assert (diffs == 1).all()


def test_montecarlo_rejects_invalid_method_and_params():
    with pytest.raises(ValueError):
        MonteCarloSimulator(method="not_a_method")
    with pytest.raises(ValueError):
        MonteCarloSimulator(block_size=0)
    with pytest.raises(ValueError):
        MonteCarloSimulator().run(_toy_returns(10), n_sims=0)


def test_montecarlo_summary_returns_monotonic_percentile_bands_and_drawdown_stats():
    returns = _toy_returns(120)
    sim = MonteCarloSimulator(method="block_bootstrap", block_size=15, random_state=2)
    curves = sim.run(returns, n_sims=200)
    summary = sim.summary(curves)

    bands = summary["percentile_bands"]
    assert list(bands.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert len(bands) == len(returns)
    # her zaman noktasinda persentiller monoton sirali olmalidir.
    assert (bands["p5"] <= bands["p25"]).all()
    assert (bands["p25"] <= bands["p50"]).all()
    assert (bands["p50"] <= bands["p75"]).all()
    assert (bands["p75"] <= bands["p95"]).all()

    assert summary["max_drawdown_mean"] <= 0.0
    assert summary["max_drawdown_p95_worst_case"] <= summary["max_drawdown_mean"]


# --------------------------------------------------------------------------
# sensitivity.py
# --------------------------------------------------------------------------


def _toy_backtest(a: float = 1.0, b: float = 1.0) -> float:
    """Sentetik backtest: 'a' performansi neredeyse degistirmez (plato),
    'b' performansi dramatik sekilde degistirir (ucurum)."""
    return 1.0 + 0.001 * a - 5.0 * (b - 1.0) ** 2


_PARAM_GRID = {"a": [0.5, 1.0, 1.5, 2.0], "b": [0.5, 1.0, 1.5, 2.0]}
_BASE_PARAMS = {"a": 1.0, "b": 1.0}


def test_parameter_sensitivity_builds_expected_long_format_dataframe():
    df = parameter_sensitivity(_toy_backtest, _PARAM_GRID, _BASE_PARAMS)

    assert list(df.index.names) == ["param_name", "param_value"]
    assert list(df.columns) == ["score"]
    assert len(df) == 8  # 2 parametre x 4 deger

    assert df.loc[("a", 1.0), "score"] == pytest.approx(_toy_backtest(a=1.0, b=1.0))
    assert df.loc[("b", 2.0), "score"] == pytest.approx(_toy_backtest(a=1.0, b=2.0))


def test_parameter_sensitivity_keeps_other_params_fixed_at_base_values():
    df = parameter_sensitivity(_toy_backtest, _PARAM_GRID, _BASE_PARAMS)
    # 'a' taranirken 'b' her zaman base_params['b']=1.0'da sabit kalmali.
    a_scores = df.xs("a", level="param_name")["score"]
    expected = [_toy_backtest(a=v, b=1.0) for v in _PARAM_GRID["a"]]
    assert a_scores.to_numpy() == pytest.approx(expected)


def test_classify_sensitivity_detects_plateau_and_cliff():
    df = parameter_sensitivity(_toy_backtest, _PARAM_GRID, _BASE_PARAMS)
    assert classify_sensitivity(df, "a") == "plateau"
    assert classify_sensitivity(df, "b") == "cliff"


def test_classify_sensitivity_rejects_unknown_param_name():
    df = parameter_sensitivity(_toy_backtest, _PARAM_GRID, _BASE_PARAMS)
    with pytest.raises(ValueError):
        classify_sensitivity(df, "not_a_param")
