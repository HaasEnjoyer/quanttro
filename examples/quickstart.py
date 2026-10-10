"""quanttro hizli baslangic ornegi: THYAO uzerinde ucdan uca akis.

Adimlar:
  1) VERI:      THYAO icin gunluk OHLCV verisini al.
  2) OZELLIK:   Basit getiri/hacim/volatilite ozellikleri uret.
  3) BACKTEST:  Cok basit bir yon (long/flat) stratejisini walk-forward mantigiyla sina.
  4) ANOMALI + RAPOR: quanttro.diagnostics ile tavan/taban anomalilerini tespit et,
     quanttro.reports ile okunabilir bir ozet rapor dosyasi uret.

Bu depo (bist-backtest) icinde quanttro paketi, bu script yazildigi sirada birkac
gelistirici tarafindan AYNI ANDA gelistiriliyordu:
  - quanttro.data      -> veri cekme/yukleme (baska bir gelistirici yaziyor)
  - quanttro.features  -> ozellik muhendisligi (baska bir gelistirici yaziyor)
  - quanttro.backtest  -> backtest motoru (baska bir gelistirici yaziyor)
  - quanttro.diagnostics / quanttro.reports -> BU oturumda tamamlandi (bu dosyanin yazari)

Bu yuzden 1-3. adimlar, nihai quanttro API'si varsa onu kullanmayi DENER; henuz
mevcut degilse (ImportError), repodaki mevcut legacy script'lerin (fetch_data.py,
feature_engineering.py) daha once uretmis oldugu CSV dosyalarina otomatik duser - boylece
bu script HER DURUMDA baştan sona calisir. 4. adim (diagnostics + reports) her zaman
quanttro paketinin kendisini kullanir, cunku o kisim bu oturumda tamamlanmistir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "THYAO"
ROOT = Path(__file__).resolve().parent.parent

# Paket henuz pip install edilmemis olabilir (repo kendi icinde gelistiriliyor) -
# bu yuzden repo kokunu dogrudan sys.path'e ekliyoruz ki `import quanttro` calissin.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def step1_get_price_data() -> pd.DataFrame:
    """Adim 1: THYAO icin gunluk OHLCV verisi.

    `quanttro.data.fetch_any` ile yfinance uzerinden canli veri cekmeyi dener
    (bkz. quanttro/data/fetch.py). Ag erisimi yoksa (ornek: offline/sandbox ortam)
    ya da modul beklenmedik bir sekilde degisirse, repodaki data/daily/THYAO.csv
    dosyasina (fetch_data.py'nin daha once uretmis oldugu veri) duser - boylece bu
    script her durumda calisir.
    """
    try:
        from quanttro.data import fetch_any

        df = fetch_any(f"{SYMBOL}.IS", interval="1d", period="max")
        if df is None or df.empty:
            raise RuntimeError("fetch_any bos sonuc dondurdu (ag erisimi olmayabilir)")
        print(f"[1/4] quanttro.data.fetch_any ile veri cekildi: {len(df)} satir")
        return df
    except Exception as exc:
        csv_path = ROOT / "data" / "daily" / f"{SYMBOL}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"quanttro.data kullanilamadi ({exc!r}) ve {csv_path} da bulunamadi. "
                "Once `python fetch_data.py` calistirin."
            ) from exc
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(
            f"[1/4] quanttro.data.fetch_any kullanilamadi ({exc!r}), mevcut CSV "
            f"kullanildi: {len(df)} satir ({csv_path.relative_to(ROOT)})"
        )
        return df


def step2_build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Adim 2: Getiri, hacim z-skoru, volatilite, ATR, tavan/taban flag'i, ileri getiri.

    `quanttro.features.core`'daki granuler fonksiyonlari (her biri tek bir
    ozellik/kolon uretir) birlestirerek bir ozellik tablosu kurar - paketin
    "kucuk, test edilebilir fonksiyonlari kendin birlestir" tasarimini gosterir.
    Modul kullanilamazsa repodaki data/features/daily/THYAO.csv dosyasina
    (feature_engineering.py ciktisi) duser.
    """
    try:
        from quanttro.features import (
            at_price_limit,
            average_true_range,
            forward_return,
            lagged_returns,
            realized_volatility,
            volume_zscore,
        )

        close, volume = price_df["Close"], price_df["Volume"]
        high, low = price_df["High"], price_df["Low"]

        feats = lagged_returns(close)  # ret_1, ret_5, ret_10, ret_20 kolonlari
        feats["volume_zscore"] = volume_zscore(volume)
        feats["realized_vol_10"] = realized_volatility(feats["ret_1"], window=10)
        feats["atr_14"] = average_true_range(high, low, close)
        feats["at_limit"] = at_price_limit(feats["ret_1"])
        feats["target_fwd_ret"] = forward_return(close, horizon=1, clip=0.12)
        feats = feats.dropna()

        print(f"[2/4] quanttro.features ile {feats.shape[1]} ozellik uretildi ({len(feats)} satir)")
        return feats
    except Exception as exc:
        csv_path = ROOT / "data" / "features" / "daily" / f"{SYMBOL}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"quanttro.features kullanilamadi ({exc!r}) ve {csv_path} da bulunamadi. "
                "Once `python feature_engineering.py` calistirin."
            ) from exc
        feats = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(
            f"[2/4] quanttro.features kullanilamadi ({exc!r}), mevcut CSV kullanildi: "
            f"{feats.shape[1]} ozellik ({csv_path.relative_to(ROOT)})"
        )
        return feats


def step3_simple_backtest(features: pd.DataFrame) -> pd.DataFrame:
    """Adim 3: Gercek quanttro.backtest API'siyle kucuk bir walk-forward deneme.

    5 ardisik (genisleyen pencere) fold'da bir LightGBM regresyon modeli egitilir,
    her fold'da yon dogrulugu ve Sharpe orani raporlanir. Bu bir alpha iddiasi
    degil - sadece WalkForwardSplitter + make_model + metrics ucgeninin nasil
    bir arada kullanildigini gosteren bir iskelet.
    """
    from quanttro.backtest import WalkForwardSplitter, make_model, sharpe_ratio

    feature_cols = ["ret_1", "ret_5", "ret_10", "ret_20", "volume_zscore", "realized_vol_10", "atr_14"]
    X, y = features[feature_cols], features["target_fwd_ret"]

    splitter = WalkForwardSplitter.expanding_window(n_folds=5)
    rows = []
    for fold in splitter.split(features.index):
        train_cutoff, test_start, test_end = fold  # Fold 3-elemanli tuple gibi unpack edilebilir
        train_mask = features.index <= train_cutoff
        test_mask = (features.index >= test_start) & (features.index <= test_end)
        if train_mask.sum() < 200 or test_mask.sum() < 20:
            continue

        model = make_model(task="regression", model_type="lightgbm")
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        actual = y[test_mask]

        strat_ret = np.sign(pred) * actual
        rows.append({
            "test_start": test_start.date(), "test_end": test_end.date(),
            "n_test": int(test_mask.sum()),
            "yon_dogrulugu": float((np.sign(pred) == np.sign(actual)).mean()),
            "sharpe": sharpe_ratio(strat_ret),
        })

    summary = pd.DataFrame(rows)
    print(f"[3/4] quanttro.backtest ile {len(summary)} walk-forward fold calistirildi")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return summary


def step4_anomaly_report_and_diagnostics(price_df: pd.DataFrame) -> None:
    """Adim 4: Tavan/taban anomali tespiti + otomatik rapor.

    Bu adim her zaman calisir - quanttro.diagnostics ve quanttro.reports bu
    oturumda tamamlanmis modullerdir (bkz. quanttro/diagnostics/, quanttro/reports/).
    """
    from quanttro.diagnostics.anomalies import detect_limit_streaks, limit_frequency_by_year
    from quanttro.reports.generate import generate_symbol_report

    returns = price_df["Close"].pct_change() if "Close" in price_df.columns else price_df["close"].pct_change()

    streaks = detect_limit_streaks(returns)
    yearly = limit_frequency_by_year(returns)

    print(f"[4/4] {SYMBOL}: {len(streaks)} ardisik tavan/taban serisi tespit edildi "
          f"(en uzunu: {streaks['length'].max() if not streaks.empty else 0} gun)")
    print(yearly.tail(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out_path = ROOT / "examples" / "output" / f"{SYMBOL}_anomaly_report.md"
    generate_symbol_report(SYMBOL, price_df, str(out_path), format="markdown")
    print(f"[4/4] Rapor yazildi: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    price_df = step1_get_price_data()
    features = step2_build_features(price_df)
    step3_simple_backtest(features)
    step4_anomaly_report_and_diagnostics(price_df)
