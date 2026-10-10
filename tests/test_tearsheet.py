"""quanttro.reports.tearsheet icin birim testleri.

`tearsheet()` testleri ozellikle ONEMLIDIR: `quanttro.backtest.metrics`
modulu bu proje uzerinde paralel olarak genisletiliyor (omega_ratio,
alpha_beta, value_at_risk vb. eklenebilir) - bu testler, o fonksiyonlar
HENUZ mevcut olmasa bile `tearsheet()`'in hatasiz calismasini dogrular
(getattr/try-except korumasinin gercekten calistigini kanitlar).

Calistirma::

    python -m pytest tests/test_tearsheet.py -v
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from quanttro.reports.tearsheet import (
    drawdown_periods,
    monthly_returns_table,
    rolling_sharpe,
    tearsheet,
)

# --------------------------------------------------------------------------
# monthly_returns_table
# --------------------------------------------------------------------------


def test_monthly_returns_table_pivots_correctly():
    # 2 ay, her ay icin bilinen gunluk getiriler -> bilesik ay getirisi elle hesaplanir.
    idx = pd.to_datetime(
        ["2023-01-02", "2023-01-03", "2023-01-04", "2023-02-01", "2023-02-02"]
    )
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02], index=idx)

    table = monthly_returns_table(returns)

    assert list(table.index) == [2023]
    expected_jan = (1.01 * 1.02 * 0.99) - 1.0
    expected_feb = (1.03 * 0.98) - 1.0
    assert table.loc[2023, 1] == pytest.approx(expected_jan)
    assert table.loc[2023, 2] == pytest.approx(expected_feb)
    # Veri olmayan aylar NaN olmali.
    assert pd.isna(table.loc[2023, 3])


def test_monthly_returns_table_multiple_years():
    idx = pd.to_datetime(["2022-05-01", "2022-05-02", "2023-05-01"])
    returns = pd.Series([0.01, 0.01, -0.01], index=idx)

    table = monthly_returns_table(returns)

    assert set(table.index) == {2022, 2023}
    assert table.loc[2022, 5] == pytest.approx(1.01 * 1.01 - 1.0)
    assert table.loc[2023, 5] == pytest.approx(-0.01)


def test_monthly_returns_table_empty_series():
    table = monthly_returns_table(pd.Series(dtype=float))
    assert table.empty


# --------------------------------------------------------------------------
# rolling_sharpe
# --------------------------------------------------------------------------


def test_rolling_sharpe_produces_finite_values_after_warmup():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    returns = pd.Series(rng.normal(0.001, 0.01, size=300), index=idx)

    result = rolling_sharpe(returns, window=50, periods_per_year=252)

    assert len(result) == len(returns)
    # Pencere oncesi (ilk window-1) NaN olmali.
    assert result.iloc[: 50 - 1].isna().all()
    # Pencere sonrasi degerler makul (sonlu, asiri olmayan) bir aralikta olmali.
    warmed_up = result.iloc[50:]
    assert warmed_up.notna().all()
    assert warmed_up.abs().max() < 50  # gunluk veriyle yillliklandirilmis Sharpe icin makul bir ust sinir


def test_rolling_sharpe_matches_manual_calculation_at_single_point():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.008, -0.012, 0.02, 0.01, -0.003], index=idx)
    window = 5

    result = rolling_sharpe(returns, window=window, periods_per_year=252)

    window_slice = returns.iloc[5 - window : 5]
    expected = window_slice.mean() / window_slice.std(ddof=1) * np.sqrt(252)
    assert result.iloc[4] == pytest.approx(expected)


def test_rolling_sharpe_constant_returns_window_is_nan():
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    returns = pd.Series([0.01] * 20, index=idx)
    result = rolling_sharpe(returns, window=10, periods_per_year=252)
    # Sabit getirili pencerelerde std=0 -> NaN olmali (bolme hatasi degil).
    assert result.iloc[10:].isna().all()


# --------------------------------------------------------------------------
# drawdown_periods
# --------------------------------------------------------------------------


def test_drawdown_periods_finds_known_decline_and_recovery():
    # Elle kurulmus senaryo: fiyat 100 -> tepe 110 (gun 2) -> dip 88 (gun 5)
    # -> tekrar 110'u asip 115'e toparlanir (gun 8).
    idx = pd.date_range("2020-01-01", periods=9, freq="B")
    prices = pd.Series([100, 105, 110, 100, 95, 88, 95, 105, 115], index=idx, dtype=float)
    returns = prices.pct_change().dropna()
    # returns index'i prices[1:]'e karsilik gelir (idx[1:]).

    table = drawdown_periods(returns, top_n=5)

    assert not table.empty
    worst = table.iloc[0]
    # En derin nokta 110 -> 88 arasi: 88/110 - 1 = -0.2
    assert worst["depth"] == pytest.approx(88.0 / 110.0 - 1.0)
    # Dip, fiyatin 88 oldugu gune denk gelmeli.
    assert worst["valley"] == idx[5]
    # Baslangic (tepe), fiyatin 110 oldugu gun olmali.
    assert worst["start"] == idx[2]
    # Toparlanma, fiyatin tekrar >=110 oldugu ilk gun olmali (105 degil, 115).
    assert worst["recovery"] == idx[8]


def test_drawdown_periods_no_recovery_when_series_ends_underwater():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.Series([100, 110, 90, 80, 85], index=idx, dtype=float)
    returns = prices.pct_change().dropna()

    table = drawdown_periods(returns, top_n=5)

    assert not table.empty
    worst = table.iloc[0]
    assert worst["recovery"] is None


def test_drawdown_periods_orders_by_depth_descending_severity():
    idx = pd.date_range("2020-01-01", periods=13, freq="B")
    # Kucuk bir dusus (tepe 100 -> dip 95, %5) sonra toparlanma,
    # ardindan daha derin bir dusus (tepe 105 -> dip 80, ~%24).
    prices = pd.Series(
        [100, 100, 95, 100, 105, 105, 95, 90, 80, 90, 100, 110, 110],
        index=idx,
        dtype=float,
    )
    returns = prices.pct_change().dropna()

    table = drawdown_periods(returns, top_n=5)

    assert len(table) >= 2
    # En derin (en negatif) ilk satirda olmali.
    assert table.iloc[0]["depth"] <= table.iloc[1]["depth"]


def test_drawdown_periods_empty_series():
    table = drawdown_periods(pd.Series(dtype=float))
    assert table.empty


# --------------------------------------------------------------------------
# tearsheet() - metrics.py henuz genislemediginde de HATASIZ calismali
# --------------------------------------------------------------------------


def _sample_returns(n: int = 500, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0006, 0.015, size=n), index=idx)


def test_tearsheet_html_runs_without_error_and_is_nonempty():
    returns = _sample_returns()
    report = tearsheet(returns, format="html")
    assert isinstance(report, str)
    assert len(report) > 0
    assert "<html" in report.lower()
    assert "Sharpe" in report


def test_tearsheet_markdown_runs_without_error_and_is_nonempty():
    returns = _sample_returns()
    report = tearsheet(returns, format="markdown")
    assert isinstance(report, str)
    assert len(report) > 0
    assert "# Performans Karnesi" in report
    assert "Sharpe" in report


def test_tearsheet_with_benchmark_runs_without_error():
    returns = _sample_returns(seed=1)
    benchmark = _sample_returns(seed=2)
    report = tearsheet(returns, benchmark_returns=benchmark, format="markdown")
    assert isinstance(report, str)
    assert len(report) > 0
    assert "Benchmark Karsilastirmasi" in report


def test_tearsheet_does_not_crash_when_optional_metrics_are_missing(monkeypatch):
    """metrics.py'de henuz alpha_beta/omega_ratio/vb. OLMADIGI durumu
    acikca simule eder (mevcut ortamda bu fonksiyonlar zaten yok, ama bu
    test niyeti aciklikla dogrular): tearsheet modulunun kullandigi
    metrics referansinda bu isimlerin kesinlikle bulunmadigindan emin
    olup, raporun yine de basariyla uretildigini kontrol eder.
    """
    # NOT: `import quanttro.reports.tearsheet as ts_module` KULLANILMAZ -
    # `quanttro/reports/__init__.py` icindeki `from .tearsheet import tearsheet`
    # satiri, paket uzerindeki `tearsheet` ATTRIBUTE'unu (nokta-zinciri ile
    # erisilen) submodul referansindan FONKSIYONA cevirir (klasik Python
    # "ayni isimli modul/fonksiyon" tuzagi). Gercek submodulu almak icin
    # sys.modules uzerinden dogrudan erisilir.
    ts_module = sys.modules["quanttro.reports.tearsheet"]

    for name in (
        "omega_ratio",
        "tail_ratio",
        "value_at_risk",
        "conditional_value_at_risk",
        "alpha_beta",
        "capture_ratio",
        "kelly_criterion",
        "risk_of_ruin",
        "ulcer_index",
        "common_sense_ratio",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "recovery_factor",
        "gain_to_pain_ratio",
    ):
        monkeypatch.delattr(ts_module._metrics, name, raising=False)

    returns = _sample_returns()
    benchmark = _sample_returns(seed=99)
    report = tearsheet(returns, benchmark_returns=benchmark, format="html")
    assert isinstance(report, str)
    assert len(report) > 0


def test_tearsheet_writes_to_output_path(tmp_path):
    returns = _sample_returns()
    out_file = tmp_path / "nested" / "tearsheet.html"
    report = tearsheet(returns, output_path=str(out_file), format="html")
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == report


def test_tearsheet_invalid_format_raises():
    returns = _sample_returns()
    with pytest.raises(ValueError):
        tearsheet(returns, format="pdf")
