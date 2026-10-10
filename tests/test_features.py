"""
quanttro.features icin pytest testleri.

Her fonksiyon icin SENTETIK (internet gerektirmeyen) veri kullanilir ve
sonuclar -mumkun oldugunda- elle/bagimsiz olarak hesaplanan degerlerle
karsilastirilir (ornegin RSI/MACD icin bagimsiz bir recursive EMA
implementasyonu, Bollinger bant genisligi icin dogrudan formul kontrolu,
Stochastic icin elle hesaplanmis ornekler).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttro.features import calendar as cal
from quanttro.features import core
from quanttro.features import cross_sectional as cs
from quanttro.features import technical as tech


# ---------------------------------------------------------------------------
# yardimcilar
# ---------------------------------------------------------------------------

def _manual_ema(values: np.ndarray, span: int) -> np.ndarray:
    """adjust=False ewm ile AYNI recursive tanim, bagimsiz numpy implementasyonu."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _manual_wilder_avg(values: np.ndarray, window: int) -> np.ndarray:
    """RSI'nin avg_gain/avg_loss icin kullandigi alpha=1/window recursive ortalama."""
    alpha = 1.0 / window
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# core.py
# ---------------------------------------------------------------------------

class TestLaggedReturns:
    def test_ret_1_matches_pct_change(self):
        close = pd.Series([10.0, 11.0, 10.5, 12.0, 11.5])
        out = core.lagged_returns(close, lags=[1], clip=None)
        expected = close.pct_change(1)
        pd.testing.assert_series_equal(out["ret_1"], expected, check_names=False)

    def test_multiple_lags_columns(self):
        close = pd.Series(np.linspace(10, 20, 30))
        out = core.lagged_returns(close, lags=[1, 5, 10], clip=None)
        assert list(out.columns) == ["ret_1", "ret_5", "ret_10"]
        pd.testing.assert_series_equal(out["ret_5"], close.pct_change(5), check_names=False)

    def test_clip_winsorizes_extreme_moves(self):
        # buyuk bir siçrama: %100 artis -> clip=0.12 ile +-0.12'ye kirpilmali
        close = pd.Series([10.0, 20.0, 19.0])
        out = core.lagged_returns(close, lags=[1], clip=0.12)
        assert out["ret_1"].iloc[1] == pytest.approx(0.12)
        # kirpma olmadan gercek deger 1.0 (100%) olurdu
        raw = close.pct_change(1)
        assert raw.iloc[1] == pytest.approx(1.0)

    def test_clip_none_leaves_raw_value(self):
        close = pd.Series([10.0, 20.0])
        out = core.lagged_returns(close, lags=[1], clip=None)
        assert out["ret_1"].iloc[1] == pytest.approx(1.0)


class TestVolumeZscore:
    def test_matches_manual_rolling_zscore(self):
        volume = pd.Series([100.0, 120.0, 90.0, 200.0, 110.0, 95.0])
        window = 3
        out = core.volume_zscore(volume, window=window)

        # index=3 icin pencere [100,120,90] -> manuel ortalama/std (ddof=1)
        window_vals = volume.iloc[1:4].to_numpy()  # [120,90,200] -> rolling(3) at idx3 uses idx1..3
        mean = window_vals.mean()
        std = window_vals.std(ddof=1)
        expected = (volume.iloc[3] - mean) / std
        assert out.iloc[3] == pytest.approx(expected)

    def test_name(self):
        out = core.volume_zscore(pd.Series([1.0, 2.0, 3.0]))
        assert out.name == "volume_zscore"


class TestRealizedVolatility:
    def test_matches_rolling_std(self):
        returns = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01, 0.02])
        out = core.realized_volatility(returns, window=3)
        expected = returns.rolling(3).std()
        pd.testing.assert_series_equal(out, expected, check_names=False)

    def test_constant_returns_zero_vol(self):
        returns = pd.Series([0.01] * 10)
        out = core.realized_volatility(returns, window=4)
        assert out.dropna().eq(0.0).all()


class TestAverageTrueRange:
    def _constant_true_range_ohlc(self):
        # her bar icin spread (high-low) = 2 sabit; close yavasca sursun
        # ki |high-prev_close| ve |low-prev_close| hep spread'den kucuk kalsin
        # -> true range HER barda tam olarak 2 olur (ilk bar haric, o da yine 2)
        close = pd.Series([10.0, 10.5, 11.0, 11.5, 12.0])
        high = close + 1.0
        low = close - 1.0
        return high, low, close

    def test_true_range_is_constant_when_spread_dominates(self):
        high, low, close = self._constant_true_range_ohlc()
        atr = core.average_true_range(high, low, close, window=3, normalize=False)
        # idx>=2 icin pencere tamamen isinmistir ve TR hep 2 -> rolling mean = 2
        assert atr.iloc[2] == pytest.approx(2.0)
        assert atr.iloc[4] == pytest.approx(2.0)

    def test_normalize_divides_by_close(self):
        high, low, close = self._constant_true_range_ohlc()
        atr_raw = core.average_true_range(high, low, close, window=3, normalize=False)
        atr_norm = core.average_true_range(high, low, close, window=3, normalize=True)
        pd.testing.assert_series_equal(atr_norm, atr_raw / close, check_names=False)


class TestAtPriceLimit:
    def test_boundary_flagging(self):
        returns = pd.Series([0.0949, 0.095, -0.10, 0.05, -0.095])
        out = core.at_price_limit(returns, threshold=0.095)
        assert out.tolist() == [0, 1, 1, 0, 1]


class TestForwardReturn:
    def test_matches_manual_formula_and_clip(self):
        close = pd.Series([10.0, 11.0, 9.0, 12.0])
        out = core.forward_return(close, horizon=1, clip=0.12)
        # ham degerler: 0.1, -0.181818..., 0.333333..., NaN
        assert out.iloc[0] == pytest.approx(0.10)
        assert out.iloc[1] == pytest.approx(-0.12)  # -0.1818 -> clip
        assert out.iloc[2] == pytest.approx(0.12)   # 0.3333 -> clip
        assert pd.isna(out.iloc[3])

    def test_no_clip_returns_raw(self):
        close = pd.Series([10.0, 11.0])
        out = core.forward_return(close, horizon=1, clip=None)
        assert out.iloc[0] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# technical.py
# ---------------------------------------------------------------------------

class TestRSI:
    def test_monotonic_increasing_is_100(self):
        close = pd.Series(np.arange(1.0, 40.0))
        out = tech.rsi(close, window=14)
        assert out.dropna().eq(100.0).all()

    def test_monotonic_decreasing_is_0(self):
        close = pd.Series(np.arange(40.0, 1.0, -1.0))
        out = tech.rsi(close, window=14)
        assert out.dropna().eq(0.0).all()

    def test_matches_independent_recursive_implementation(self):
        rng = np.random.default_rng(42)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=60)))
        window = 14

        out = tech.rsi(close, window=window)

        delta = close.diff().to_numpy()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        # ewm(adjust=False) ilk gecerli (NaN olmayan) degerden baslar: delta[0] NaN -> gain/loss[0]=0
        gain[0] = 0.0
        loss[0] = 0.0
        avg_gain = _manual_wilder_avg(gain, window)
        avg_loss = _manual_wilder_avg(loss, window)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            expected_rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))

        # min_periods=window oncesi degerler NaN olmali
        assert out.iloc[: window - 1].isna().all()
        np.testing.assert_allclose(out.iloc[window - 1:].to_numpy(), expected_rsi[window - 1:], rtol=1e-9)

    def test_bounded_between_0_and_100(self):
        rng = np.random.default_rng(7)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 2, size=100)))
        out = tech.rsi(close, window=14).dropna()
        assert (out >= 0).all() and (out <= 100).all()


class TestMACD:
    def test_matches_independent_ema_implementation(self):
        rng = np.random.default_rng(1)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=80)))
        fast, slow, signal = 12, 26, 9

        out = tech.macd(close, fast=fast, slow=slow, signal=signal)

        values = close.to_numpy()
        ema_fast = _manual_ema(values, fast)
        ema_slow = _manual_ema(values, slow)
        macd_line = ema_fast - ema_slow
        signal_line = _manual_ema(macd_line, signal)
        histogram = macd_line - signal_line

        np.testing.assert_allclose(out["macd"].to_numpy(), macd_line, rtol=1e-9)
        np.testing.assert_allclose(out["signal"].to_numpy(), signal_line, rtol=1e-9)
        np.testing.assert_allclose(out["histogram"].to_numpy(), histogram, rtol=1e-9)

    def test_histogram_equals_macd_minus_signal(self):
        close = pd.Series(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 50)))
        out = tech.macd(close)
        pd.testing.assert_series_equal(
            out["histogram"], out["macd"] - out["signal"], check_names=False,
        )


class TestBollingerBands:
    def test_band_width_matches_formula(self):
        rng = np.random.default_rng(3)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, size=50)))
        window, n_std = 20, 2.0
        out = tech.bollinger_bands(close, window=window, n_std=n_std)

        std = close.rolling(window).std()
        width = out["upper"] - out["lower"]
        np.testing.assert_allclose(width.dropna().to_numpy(), (2 * n_std * std).dropna().to_numpy(), rtol=1e-9)

    def test_known_window_matches_independent_numpy_calculation(self):
        values = [100, 102, 101, 103, 99, 98, 104, 105, 97, 100,
                   101, 103, 102, 99, 98, 100, 101, 102, 103, 104]
        close = pd.Series([float(v) for v in values])
        out = tech.bollinger_bands(close, window=20, n_std=2.0)

        mean = np.mean(values)
        std = np.std(values, ddof=1)  # pandas .std() varsayilani ddof=1 (ornek std)
        expected_upper = mean + 2 * std
        expected_lower = mean - 2 * std
        expected_pb = (values[-1] - expected_lower) / (expected_upper - expected_lower)

        last = out.iloc[-1]
        assert last["middle"] == pytest.approx(mean)
        assert last["upper"] == pytest.approx(expected_upper)
        assert last["lower"] == pytest.approx(expected_lower)
        assert last["%b"] == pytest.approx(expected_pb)

    def test_zero_variance_window_collapses_bands(self):
        close = pd.Series([100.0] * 20)
        out = tech.bollinger_bands(close, window=20, n_std=2.0)
        # std=0 -> upper=lower=middle (bant genisligi 0)
        assert out["upper"].iloc[-1] == pytest.approx(out["lower"].iloc[-1])
        assert out["upper"].iloc[-1] == pytest.approx(100.0)


class TestStochasticOscillator:
    def test_hand_calculated_percent_k(self):
        high = pd.Series([10.0, 12.0, 11.0, 13.0, 9.0])
        low = pd.Series([8.0, 9.0, 10.0, 9.0, 7.0])
        close = pd.Series([9.0, 11.0, 10.5, 12.0, 8.0])
        out = tech.stochastic_oscillator(high, low, close, window=3, d_window=3)

        # idx=2: pencere [0,1,2] -> HH=max(10,12,11)=12, LL=min(8,9,10)=8, close=10.5
        assert out["%K"].iloc[2] == pytest.approx(100 * (10.5 - 8) / (12 - 8))
        # idx=3: pencere [1,2,3] -> HH=max(12,11,13)=13, LL=min(9,10,9)=9, close=12
        assert out["%K"].iloc[3] == pytest.approx(100 * (12 - 9) / (13 - 9))
        # idx=4: pencere [2,3,4] -> HH=max(11,13,9)=13, LL=min(10,9,7)=7, close=8
        assert out["%K"].iloc[4] == pytest.approx(100 * (8 - 7) / (13 - 7))

        # %D = %K'nin 3 barlik SMA'si -> idx=4'te ilk gecerli deger
        expected_d = out["%K"].iloc[2:5].mean()
        assert out["%D"].iloc[4] == pytest.approx(expected_d)

    def test_percent_k_at_window_extremes(self):
        # close pencerenin en yuksegine esitse %K=100, en dusugune esitse %K=0
        high = pd.Series([10.0, 20.0, 15.0])
        low = pd.Series([5.0, 8.0, 7.0])
        close_at_high = pd.Series([10.0, 20.0, 20.0])
        out_high = tech.stochastic_oscillator(high, low, close_at_high, window=3)
        assert out_high["%K"].iloc[2] == pytest.approx(100.0)

        close_at_low = pd.Series([10.0, 20.0, 5.0])
        out_low = tech.stochastic_oscillator(high, low, close_at_low, window=3)
        assert out_low["%K"].iloc[2] == pytest.approx(0.0)


class TestOnBalanceVolume:
    def test_hand_calculated_obv(self):
        close = pd.Series([10.0, 11.0, 11.0, 9.0, 9.0, 12.0])
        volume = pd.Series([100.0, 200.0, 150.0, 300.0, 120.0, 80.0])
        out = tech.on_balance_volume(close, volume)
        # direction: [0, +1, 0, -1, 0, +1] -> contrib: [0,200,0,-300,0,80]
        # cumsum: [0, 200, 200, -100, -100, -20]
        expected = [0.0, 200.0, 200.0, -100.0, -100.0, -20.0]
        np.testing.assert_allclose(out.to_numpy(), expected)

    def test_monotonic_up_obv_increments_match_volume(self):
        close = pd.Series(np.arange(1.0, 10.0))
        volume = pd.Series(np.full(9, 50.0))
        out = tech.on_balance_volume(close, volume)
        diffs = out.diff().dropna()
        assert diffs.eq(50.0).all()


class TestMomentum:
    def test_matches_diff(self):
        close = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0])
        out = tech.momentum(close, window=2)
        expected = close.diff(2)
        pd.testing.assert_series_equal(out, expected, check_names=False)


# ---------------------------------------------------------------------------
# calendar.py
# ---------------------------------------------------------------------------

class TestDayOfWeekFeatures:
    def test_one_hot_matches_day_of_week(self):
        index = pd.date_range("2024-01-01", periods=7, freq="D")  # Pzt..Paz
        out = cal.day_of_week_features(index)

        assert list(out["day_of_week"]) == [0, 1, 2, 3, 4, 5, 6]
        for i, dow in enumerate(out["day_of_week"]):
            for d in range(7):
                expected = 1 if d == dow else 0
                assert out[f"dow_{d}"].iloc[i] == expected

    def test_all_seven_columns_always_present(self):
        # sadece is gunleri (Pzt-Cuma) iceren bir index verilse bile dow_5/dow_6 kolonlari olmali
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        out = cal.day_of_week_features(index)
        for d in range(7):
            assert f"dow_{d}" in out.columns
        assert out["dow_5"].eq(0).all() and out["dow_6"].eq(0).all()


class TestTurnOfMonthFlag:
    def test_30_day_month_default_window(self):
        index = pd.date_range("2024-04-01", "2024-04-30", freq="D")  # Nisan: 30 gun
        out = cal.turn_of_month_flag(index, n_days=3)

        flagged_days = {1, 2, 3, 28, 29, 30}
        for day in range(1, 31):
            expected = 1 if day in flagged_days else 0
            assert out.iloc[day - 1] == expected, f"gun {day} icin beklenmeyen flag"

    def test_custom_n_days(self):
        index = pd.date_range("2024-02-01", "2024-02-29", freq="D")  # 2024 artik yil: 29 gun
        out = cal.turn_of_month_flag(index, n_days=1)
        assert out.iloc[0] == 1    # gun 1
        assert out.iloc[-1] == 1   # gun 29
        assert out.iloc[1] == 0    # gun 2
        assert out.iloc[-2] == 0   # gun 28


class TestHourOfDayFeatures:
    def test_matches_index_hour(self):
        index = pd.date_range("2024-01-02 09:00", periods=5, freq="h")
        out = cal.hour_of_day_features(index)
        assert out.tolist() == [9, 10, 11, 12, 13]


class TestIsMondayOrTuesday:
    def test_flags_only_monday_tuesday(self):
        index = pd.date_range("2024-01-01", periods=7, freq="D")  # Pzt..Paz
        out = cal.is_monday_or_tuesday(index)
        assert out.tolist() == [1, 1, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# cross_sectional.py
# ---------------------------------------------------------------------------

class TestRelativeStrength:
    def test_matches_manual_demeaning(self):
        df = pd.DataFrame({
            "A": [0.01, -0.02, 0.03],
            "B": [0.02, 0.00, -0.01],
            "C": [-0.01, 0.01, 0.02],
        })
        out = cs.relative_strength(df)
        market_mean = df.mean(axis=1)
        for col in df.columns:
            expected = df[col] - market_mean
            pd.testing.assert_series_equal(out[col], expected, check_names=False)

    def test_rows_sum_to_zero(self):
        df = pd.DataFrame({
            "A": [0.05, 0.10],
            "B": [-0.02, 0.00],
            "C": [0.01, -0.05],
        })
        out = cs.relative_strength(df)
        np.testing.assert_allclose(out.sum(axis=1).to_numpy(), [0.0, 0.0], atol=1e-12)


class TestRelativeForwardTarget:
    def test_matches_manual_demeaning(self):
        df = pd.DataFrame({
            "A": [0.03, 0.01],
            "B": [0.05, -0.02],
            "C": [0.02, 0.00],
        })
        out = cs.relative_forward_target(df)
        market_mean = df.mean(axis=1)
        for col in df.columns:
            expected = df[col] - market_mean
            pd.testing.assert_series_equal(out[col], expected, check_names=False)


class TestCrossSectionalRank:
    def test_known_percentile_ranks(self):
        df = pd.DataFrame({
            "A": [10.0, 5.0],
            "B": [30.0, 5.0],
            "C": [20.0, 15.0],
        })
        out = cs.cross_sectional_rank(df)
        # satir 0: A=10(1.)/3, C=20(2.)/3, B=30(3.)/3
        assert out["A"].iloc[0] == pytest.approx(1 / 3)
        assert out["C"].iloc[0] == pytest.approx(2 / 3)
        assert out["B"].iloc[0] == pytest.approx(3 / 3)
        # satir 1: A ve B esit (5.0) -> ortalama rank 1.5/3; C en yuksek -> 3/3
        assert out["A"].iloc[1] == pytest.approx(1.5 / 3)
        assert out["B"].iloc[1] == pytest.approx(1.5 / 3)
        assert out["C"].iloc[1] == pytest.approx(1.0)

    def test_bounded_between_0_and_1(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.normal(size=(10, 6)), columns=list("ABCDEF"))
        out = cs.cross_sectional_rank(df)
        assert (out.to_numpy() > 0).all() and (out.to_numpy() <= 1.0).all()


class TestSectorRelativeStrength:
    def test_no_sector_map_falls_back_to_market(self):
        df = pd.DataFrame({
            "A": [0.02, 0.01],
            "B": [0.04, -0.01],
            "C": [0.01, 0.00],
            "D": [0.03, 0.02],
        })
        out_none = cs.sector_relative_strength(df, sector_map=None)
        out_market = cs.relative_strength(df)
        pd.testing.assert_frame_equal(out_none, out_market)

    def test_sector_grouping_with_fallback_for_unmapped_and_singleton(self):
        # A,B -> "bank" sektoru (2 uye, gercek sektor-bazli hesap)
        # C -> "retail" sektoru ama TEK uye -> piyasaya fallback
        # D -> sector_map'te YOK -> piyasaya fallback
        df = pd.DataFrame({
            "A": [0.02],
            "B": [0.04],
            "C": [0.01],
            "D": [0.03],
        })
        sector_map = {"A": "bank", "B": "bank", "C": "retail"}
        out = cs.sector_relative_strength(df, sector_map=sector_map)

        market_mean = df.mean(axis=1).iloc[0]  # (0.02+0.04+0.01+0.03)/4 = 0.025
        bank_mean = df[["A", "B"]].mean(axis=1).iloc[0]  # (0.02+0.04)/2 = 0.03

        assert out["A"].iloc[0] == pytest.approx(0.02 - bank_mean)
        assert out["B"].iloc[0] == pytest.approx(0.04 - bank_mean)
        assert out["C"].iloc[0] == pytest.approx(0.01 - market_mean)
        assert out["D"].iloc[0] == pytest.approx(0.03 - market_mean)

    def test_output_columns_match_input_order(self):
        df = pd.DataFrame({
            "Z": [0.01, 0.02],
            "A": [0.03, 0.01],
            "M": [0.02, 0.00],
        })
        out = cs.sector_relative_strength(df, sector_map={"Z": "s1", "A": "s1"})
        assert list(out.columns) == list(df.columns)
