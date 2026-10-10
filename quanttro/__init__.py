"""quanttro: acik kaynak kantitatif arastirma kutuphanesi (backtest robustness,
risk/portfolio, prop-firm sinav simulasyonu) - herhangi bir yfinance ticker'i
ile calisir, Borsa Istanbul (BIST) icin yerlesik (built-in) destek icerir.

Hizli baslangic
---------------
    import quanttro as bq

    df = bq.fetch("THYAO.IS")                 # tek hisse -> DataFrame
    data = bq.fetch("BIST30")                  # tum BIST30 -> {ticker: DataFrame}

    feats = bq.make_features(df)               # core + technical + calendar ozellikleri
    issues = bq.check(df)                       # veri kalitesi taramasi
    bq.report("THYAO.IS", df, "thyao.md")       # otomatik ozet rapor

    splitter = bq.WalkForwardSplitter(n_folds=5)
    model = bq.make_model(task="regression")

Bu modul, alt paketlerin (data/features/backtest/diagnostics/reports) en sik
kullanilan fonksiyonlarini tek bir ust-duzey arayuzde toplayan bir "facade"
katmanidir - her alt paketi ayri ayri import etmeye gerek kalmadan hizli
baslamak icin. Daha ince kontrol/az kullanilan fonksiyonlar icin ilgili alt
paketi (ornegin ``quanttro.features.cross_sectional``) dogrudan import edin.
"""

from __future__ import annotations

import pandas as pd

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Alt paketler (dogrudan erisim icin: bq.data, bq.features, bq.backtest, ...)
# ---------------------------------------------------------------------------
from quanttro import backtest, data, diagnostics, features, live, portfolio, propfirm, reports, risk, robustness

# ---------------------------------------------------------------------------
# En sik kullanilan isimlerin ust-duzeyde (flat) yeniden-disa-aktarimi
# ---------------------------------------------------------------------------
from quanttro.backtest import (
    Fold,
    ModelFactory,
    WalkForwardSplitter,
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
    information_coefficient,
    kelly_criterion,
    make_model,
    max_drawdown,
    omega_ratio,
    payoff_ratio,
    profit_factor,
    recovery_factor,
    risk_of_ruin,
    rolling_ic,
    sharpe_ratio,
    sortino_ratio,
    tail_ratio,
    tertile_spread,
    ulcer_index,
    value_at_risk,
)
from quanttro.data import (
    BIST30,
    BIST100,
    SymbolSet,
    fetch_any,
    fetch_macro,
    fetch_ohlcv,
    resolve_symbols,
    validation_report,
)
from quanttro.diagnostics import (
    detect_limit_streaks,
    detect_regime_break,
    flag_outlier_stocks,
    limit_frequency_by_year,
)
from quanttro.features.calendar import (
    day_of_week_features,
    is_monday_or_tuesday,
    turn_of_month_flag,
)
from quanttro.features.core import (
    at_price_limit,
    average_true_range,
    forward_return,
    lagged_returns,
    realized_volatility,
    volume_zscore,
)
from quanttro.features.technical import bollinger_bands, macd, on_balance_volume, rsi
from quanttro.portfolio import (
    compare_strategies,
    covariance_shrinkage,
    discrete_allocation,
    efficient_risk,
    hrp_optimize,
    mean_variance_optimize,
    rebase,
)
from quanttro.live import BrokerInterface, LiveTradingLoop, Order, SimulatedBroker
from quanttro.propfirm import (
    FTMO_CHALLENGE_RULES,
    TOPSTEP_COMBINE_RULES,
    TRAILING_ONLY_NO_DAILY_LIMIT_RULES,
    PropFirmResult,
    PropFirmRules,
    simulate_prop_firm_challenge,
)
from quanttro.reports import generate_symbol_report
from quanttro.reports.tearsheet import drawdown_periods, monthly_returns_table, rolling_sharpe, tearsheet
from quanttro.risk import (
    DrawdownThrottle,
    OrderFill,
    aggregate_portfolio_drawdown,
    PositionSizer,
    TransactionCostModel,
    apply_stop_loss_take_profit,
    atr_based_size,
    fixed_fractional,
    kelly_fraction,
    realistic_slippage,
    simulate_partial_fills,
    volatility_target_size,
)
from quanttro.robustness import (
    CombinatorialPurgedCV,
    MonteCarloSimulator,
    classify_sensitivity,
    parameter_sensitivity,
    probability_of_backtest_overfitting,
)

__all__ = [
    "__version__",
    # alt paketler
    "data", "features", "backtest", "diagnostics", "reports", "risk", "portfolio", "robustness", "live", "propfirm",
    # ust-duzey kolaylik fonksiyonlari (bu dosyada tanimli)
    "fetch", "make_features", "check", "report",
    # data
    "BIST30", "BIST100", "SymbolSet", "resolve_symbols",
    "fetch_any", "fetch_ohlcv", "fetch_macro", "validation_report",
    # features
    "lagged_returns", "volume_zscore", "realized_volatility", "average_true_range",
    "at_price_limit", "forward_return", "rsi", "macd", "bollinger_bands",
    "on_balance_volume", "day_of_week_features", "turn_of_month_flag", "is_monday_or_tuesday",
    # backtest - temel
    "Fold", "WalkForwardSplitter", "ModelFactory", "make_model",
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "hit_rate",
    "profit_factor", "deflated_sharpe_ratio", "information_coefficient",
    "tertile_spread", "rolling_ic",
    # backtest - genisletilmis risk/performans metrikleri
    "alpha_beta", "omega_ratio", "tail_ratio", "value_at_risk", "conditional_value_at_risk",
    "capture_ratio", "avg_win", "avg_loss", "payoff_ratio", "kelly_criterion",
    "risk_of_ruin", "ulcer_index", "common_sense_ratio", "recovery_factor", "gain_to_pain_ratio",
    # diagnostics
    "detect_limit_streaks", "limit_frequency_by_year", "flag_outlier_stocks",
    "detect_regime_break",
    # reports
    "generate_symbol_report", "tearsheet", "monthly_returns_table", "rolling_sharpe", "drawdown_periods",
    # risk (pozisyon boyutlandirma / drawdown / islem maliyeti)
    "PositionSizer", "fixed_fractional", "kelly_fraction", "volatility_target_size",
    "atr_based_size", "DrawdownThrottle", "aggregate_portfolio_drawdown", "apply_stop_loss_take_profit", "TransactionCostModel",
    "realistic_slippage", "simulate_partial_fills", "OrderFill",
    # portfolio (optimizasyon / tahsis / karsilastirma)
    "mean_variance_optimize", "efficient_risk", "covariance_shrinkage", "hrp_optimize",
    "discrete_allocation", "rebase", "compare_strategies",
    # robustness (saglamlik testleri)
    "CombinatorialPurgedCV", "probability_of_backtest_overfitting", "MonteCarloSimulator",
    "parameter_sensitivity", "classify_sensitivity",
    # live (paper/canli islem koprusu)
    "BrokerInterface", "SimulatedBroker", "LiveTradingLoop", "Order",
    # propfirm (degerlendirme sinavi simulasyonu)
    "PropFirmRules", "FTMO_CHALLENGE_RULES", "TOPSTEP_COMBINE_RULES", "TRAILING_ONLY_NO_DAILY_LIMIT_RULES",
    "PropFirmResult", "simulate_prop_firm_challenge",
]


# ---------------------------------------------------------------------------
# Ust-duzey kolaylik fonksiyonlari
# ---------------------------------------------------------------------------

def fetch(symbols, interval: str = "1d", period: str = "max", **kwargs):
    """Tek bir kolaylik fonksiyonu: tek ticker, ozel ticker listesi, veya
    "BIST30"/"BIST100" kabul eder.

    - ``fetch("THYAO.IS")`` -> tek bir DataFrame (veri yoksa None)
    - ``fetch("PAXG-USD")`` -> yfinance'in tanidigi herhangi bir ticker (crypto/altin dahil)
    - ``fetch(["THYAO.IS", "AKBNK.IS"])`` -> ``{ticker: DataFrame}`` sozlugu
    - ``fetch("BIST30")`` / ``fetch("BIST100")`` -> ilgili tum hisseler icin sozluk

    Daha ince kontrol (cache ayarlari, sleep suresi vb.) icin dogrudan
    ``quanttro.data.fetch_any`` / ``fetch_ohlcv`` kullanin.
    """
    if isinstance(symbols, str) and symbols.lower() not in ("bist30", "bist100"):
        return fetch_any(symbols, interval=interval, period=period, **kwargs)
    resolved = resolve_symbols(symbols)
    return fetch_ohlcv(resolved, interval=interval, period=period, **kwargs)


def make_features(df: pd.DataFrame, include: tuple[str, ...] = ("core", "technical", "calendar")) -> pd.DataFrame:
    """Bir OHLCV DataFrame'inden (Close/High/Low/Volume kolonlu) hazir bir
    ozellik tablosu uretir.

    ``include`` ile hangi ozellik gruplarinin ekleneceginizi secebilirsiniz:

    - ``"core"``: gecikmeli getiriler, hacim z-skoru, gerceklesen volatilite,
      ATR, tavan/taban flag'i, 1-bar ileri (forward) getiri hedefi.
    - ``"technical"``: RSI(14), MACD, Bollinger Bands, OBV.
    - ``"calendar"``: haftanin gunu (one-hot), ay donumu flag'i, Pazartesi/Sali
      kurali flag'i (bu oturumda bulunan hacim-takvim etkisiyle ilgili).

    Not: bu fonksiyon TEK bir hisse icindir. Coklu-hisse (kesitsel/panel)
    ozellikleri icin (``relative_strength``, ``cross_sectional_rank`` vb.)
    ``quanttro.features.cross_sectional`` modulunu dogrudan kullanin -
    bunlar birden fazla hissenin ayni-zamanli verisini gerektirdigi icin tek
    hisseli bu fonksiyona dahil edilmemistir.
    """
    cols = {c.lower(): c for c in df.columns}
    close = df[cols["close"]] if "close" in cols else df["Close"]
    high = df[cols["high"]] if "high" in cols else None
    low = df[cols["low"]] if "low" in cols else None
    volume = df[cols["volume"]] if "volume" in cols else None

    out = pd.DataFrame(index=df.index)
    out["close"] = close

    if "core" in include:
        out = out.join(lagged_returns(close))
        if volume is not None:
            out["volume_zscore"] = volume_zscore(volume)
        ret_1 = close.pct_change()
        out["realized_vol_10"] = realized_volatility(ret_1)
        if high is not None and low is not None:
            out["atr_14"] = average_true_range(high, low, close)
        out["at_limit"] = at_price_limit(ret_1)
        out["target_fwd_ret"] = forward_return(close)

    if "technical" in include:
        out["rsi_14"] = rsi(close)
        out = out.join(macd(close).add_prefix("macd_"))
        out = out.join(bollinger_bands(close).add_prefix("bb_"))
        if volume is not None:
            out["obv"] = on_balance_volume(close, volume)

    if "calendar" in include:
        out = out.join(day_of_week_features(df.index))
        out["turn_of_month"] = turn_of_month_flag(df.index)
        out["monday_or_tuesday"] = is_monday_or_tuesday(df.index)

    return out


def check(df: pd.DataFrame) -> dict:
    """Bir OHLCV DataFrame'i veri kalitesi sorunlarina (anormal fiyat
    sicramasi, ardisik sifir hacim, donmus/sabit fiyat) karsi tarar.

    ``quanttro.data.validation_report`` icin kisayoldur. Bu oturumda
    THYAO'nun 2005-01-03 tarihli yfinance redenominasyon hatasi gibi
    sorunlari otomatik yakalamak icin kullanilir - herhangi bir backtest'e
    baslamadan once calistirilmasi onerilir.
    """
    return validation_report(df)


def report(symbol: str, df: pd.DataFrame, output_path: str, format: str = "markdown") -> None:
    """Bir sembol icin otomatik ozet rapor uretir (fiyat/hacim istatistikleri +
    anomali tespiti). ``quanttro.reports.generate_symbol_report`` icin kisayoldur."""
    return generate_symbol_report(symbol, df, output_path, format=format)
