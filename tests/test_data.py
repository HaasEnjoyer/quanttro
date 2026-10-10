"""quanttro.data icin birim testleri.

Gercek internet erisimi gerektirmez; tum testler sentetik DataFrame'ler
uzerinde calisir. Senaryolar bu arastirma oturumunda BIST verisinde
rastlanan gercek anomalilere dayanir (THYAO 2005-01-03 redenominasyon
sicramasi, ardisik sifir-hacim barlari, donmus/stale fiyatlar).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from quanttro.data.cache import CacheConfig, read_cache, write_cache
from quanttro.data.symbols import BIST30, BIST100, SymbolSet, resolve_symbols
from quanttro.data.validate import (
    detect_price_jumps,
    detect_stale_prices,
    detect_zero_volume_runs,
    validation_report,
)


def _make_ohlcv(close: list[float], volume: list[float] | None = None) -> pd.DataFrame:
    """Testler icin basit bir sentetik OHLCV DataFrame'i uretir."""
    n = len(close)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close_s = pd.Series(close, index=dates, dtype=float)
    if volume is None:
        volume = [1000] * n
    return pd.DataFrame(
        {
            "Open": close_s,
            "High": close_s * 1.01,
            "Low": close_s * 0.99,
            "Close": close_s,
            "Volume": volume,
        },
        index=dates,
    )


class TestDetectPriceJumps:
    def test_no_jump_on_smooth_series(self):
        df = _make_ohlcv([10, 10.1, 10.2, 10.0, 9.9, 10.05])
        result = detect_price_jumps(df, threshold=0.5)
        assert result.empty

    def test_thyao_style_redenomination_jump_detected(self):
        # THYAO 2005-01-03 civarinda onceki bara gore buyuk bir sicrama yapmisti
        # (redenominasyon hatasi). Burada ayni buyuklukte bir sicrama simule edilir.
        close = [0.00092, 0.0019, 0.00188]
        df = _make_ohlcv(close)
        result = detect_price_jumps(df, threshold=0.5)
        assert len(result) == 1
        assert result.index[0] == df.index[1]
        assert result["pct_change"].iloc[0] > 0.5

    def test_extreme_982x_jump_detected(self):
        close = [1.0, 982.0, 980.0]
        df = _make_ohlcv(close)
        result = detect_price_jumps(df, threshold=0.5)
        assert len(result) == 1
        assert result.index[0] == df.index[1]

    def test_downward_jump_also_detected(self):
        close = [100.0, 40.0, 41.0]
        result = detect_price_jumps(_make_ohlcv(close), threshold=0.5)
        assert len(result) == 1

    def test_threshold_respected(self):
        close = [100.0, 130.0, 131.0]  # ~%30 artis
        result_strict = detect_price_jumps(_make_ohlcv(close), threshold=0.5)
        assert result_strict.empty
        result_loose = detect_price_jumps(_make_ohlcv(close), threshold=0.25)
        assert len(result_loose) == 1


class TestDetectZeroVolumeRuns:
    def test_short_zero_run_not_flagged(self):
        df = _make_ohlcv([10] * 5, volume=[100, 0, 0, 100, 100])
        result = detect_zero_volume_runs(df, min_run=3)
        assert result.empty

    def test_long_zero_run_flagged(self):
        df = _make_ohlcv([10] * 6, volume=[100, 0, 0, 0, 0, 100])
        result = detect_zero_volume_runs(df, min_run=3)
        assert len(result) == 4
        assert (result["run_length"] == 4).all()

    def test_multiple_runs_get_distinct_ids(self):
        volume = [0, 0, 0, 100, 100, 0, 0, 0, 0]
        df = _make_ohlcv([10] * len(volume), volume=volume)
        result = detect_zero_volume_runs(df, min_run=3)
        assert result["run_id"].nunique() == 2

    def test_nan_volume_treated_as_zero(self):
        volume = [100, np.nan, np.nan, np.nan, 100]
        df = _make_ohlcv([10] * 5, volume=volume)
        result = detect_zero_volume_runs(df, min_run=3)
        assert len(result) == 3


class TestDetectStalePrices:
    def test_normal_moving_prices_not_flagged(self):
        df = _make_ohlcv([10, 10.1, 10.2, 10.3, 10.4, 10.5])
        result = detect_stale_prices(df, min_run=5)
        assert result.empty

    def test_short_repeat_not_flagged(self):
        df = _make_ohlcv([10, 10, 10, 11, 12])
        result = detect_stale_prices(df, min_run=5)
        assert result.empty

    def test_long_stale_run_flagged(self):
        df = _make_ohlcv([10, 10, 10, 10, 10, 10, 11])
        result = detect_stale_prices(df, min_run=5)
        assert len(result) == 6
        assert (result["run_length"] == 6).all()

    def test_flagged_rows_are_the_repeated_block(self):
        df = _make_ohlcv([5, 5, 5, 5, 5, 8, 9])
        result = detect_stale_prices(df, min_run=5)
        assert list(result.index) == list(df.index[:5])


class TestValidationReport:
    def test_clean_data_reports_clean(self):
        df = _make_ohlcv([10, 10.1, 10.2, 10.15, 10.3, 10.25])
        report = validation_report(df)
        assert report["is_clean"] is True
        assert report["price_jumps"]["count"] == 0
        assert report["zero_volume_runs"]["count"] == 0
        assert report["stale_price_runs"]["count"] == 0
        assert report["n_rows"] == len(df)

    def test_dirty_data_reports_all_issues(self):
        close = [1.0, 982.0] + [980.0] * 6
        volume = [100, 100] + [0] * 6
        df = _make_ohlcv(close, volume=volume)
        report = validation_report(
            df, jump_threshold=0.5, zero_volume_min_run=3, stale_min_run=5
        )
        assert report["is_clean"] is False
        assert report["price_jumps"]["count"] >= 1
        assert report["zero_volume_runs"]["count"] >= 1
        assert report["stale_price_runs"]["count"] >= 1

    def test_lowercase_columns_supported(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {"close": [10.0, 10.1, 10.2, 10.3], "volume": [100, 100, 100, 100]},
            index=dates,
        )
        report = validation_report(df)
        assert report["is_clean"] is True


class TestSymbols:
    def test_bist30_has_30_tickers(self):
        assert len(BIST30) == 30
        assert all(t.endswith(".IS") for t in BIST30)

    def test_bist100_is_superset_of_bist30_and_large_enough(self):
        assert set(BIST30).issubset(set(BIST100))
        assert len(BIST100) >= 50

    def test_resolve_symbols_by_enum(self):
        assert resolve_symbols(SymbolSet.BIST30) == BIST30
        assert resolve_symbols(SymbolSet.BIST100) == BIST100

    def test_resolve_symbols_by_string_case_insensitive(self):
        assert resolve_symbols("bist30") == BIST30
        assert resolve_symbols("BIST100") == BIST100

    def test_resolve_symbols_custom_list_passthrough(self):
        custom = ["PAXG-USD", "BTC-USD", "THYAO.IS"]
        assert resolve_symbols(custom) == custom

    def test_resolve_symbols_unknown_string_raises(self):
        with pytest.raises(ValueError):
            resolve_symbols("bist500")


class TestCache:
    def test_write_then_read_roundtrip(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path, ttl_seconds=3600)
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        write_cache(df, "raw", "TESTSYM", "1d", "max", config)
        cached = read_cache("raw", "TESTSYM", "1d", "max", config)
        assert cached is not None
        pd.testing.assert_frame_equal(cached, df)

    def test_missing_entry_returns_none(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path)
        assert read_cache("raw", "NOPE", "1d", "max", config) is None

    def test_expired_entry_returns_none(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path, ttl_seconds=0.01)
        df = pd.DataFrame({"Close": [1.0]})
        write_cache(df, "raw", "TESTSYM", "1d", "max", config)
        time.sleep(0.05)
        assert read_cache("raw", "TESTSYM", "1d", "max", config) is None

    def test_disabled_cache_always_misses(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path, enabled=False)
        df = pd.DataFrame({"Close": [1.0]})
        assert write_cache(df, "raw", "TESTSYM", "1d", "max", config) is None
        assert read_cache("raw", "TESTSYM", "1d", "max", config) is None
