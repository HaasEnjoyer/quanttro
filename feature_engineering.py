from pathlib import Path

import pandas as pd

from symbols import BIST30

DATA_DIR = Path(__file__).parent / "data"
MACRO_DIR = Path(__file__).parent / "data" / "macro"
OUT_DIR = Path(__file__).parent / "data" / "features"

LAG_BARS = [1, 5, 10, 20]
LIMIT_THRESHOLD = 0.095  # BIST %10 tavan/taban icin guvenlik payli esik
RETURN_CLIP = 0.12       # winsorize: 19 Mart 2025 gibi piyasa-geneli sok barlarini yumusat
TREND_WINDOW = 60        # USD/TRY'nin uzun vadeli enflasyon/devaluasyon trendini tahmin etmek icin pencere


def load_symbol(timeframe: str, ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / timeframe / f"{ticker}.csv"
    if not path.exists():
        return None
    index_col = "Date" if timeframe == "daily" else "Datetime"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = index_col
    return df


def build_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["Close"].pct_change().clip(-RETURN_CLIP, RETURN_CLIP)

    out["close"] = df["Close"]
    out["ret_1"] = ret
    for n in LAG_BARS[1:]:
        out[f"ret_{n}"] = df["Close"].pct_change(n).clip(-RETURN_CLIP, RETURN_CLIP)

    volume = df["Volume"]
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_zscore"] = (volume - vol_mean) / vol_std

    out["realized_vol_10"] = ret.rolling(10).std()

    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14).mean() / df["Close"]

    out["at_limit"] = (ret.abs() >= LIMIT_THRESHOLD).astype(int)

    if timeframe == "daily":
        out["day_of_week"] = df.index.dayofweek
        days_in_month = df.index.days_in_month
        day_of_month = df.index.day
        out["is_turn_of_month"] = (
            (day_of_month <= 3) | (day_of_month >= days_in_month - 2)
        ).astype(int)
    else:
        out["hour_of_day"] = df.index.hour
        out["day_of_week"] = df.index.dayofweek

    out["target_fwd_ret"] = ret.shift(-1)
    fwd5 = (df["Close"].shift(-5) / df["Close"] - 1).clip(-RETURN_CLIP * 2, RETURN_CLIP * 2)
    out["target_fwd5_ret"] = fwd5
    return out


def load_macro_aligned(timeframe: str, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    path = MACRO_DIR / timeframe / "USDTRY.csv"
    macro = pd.read_csv(path, index_col=0, parse_dates=True)
    usdtry_ret = macro["Close"].pct_change()

    # trend: uzun pencere ortalamasi (devaluasyon hizi); detrended: o anki hareketin
    # trendden sapmasi (surpriz) - ham ret_1 enflasyon trendini tasidigi icin hisse
    # getirileriyle sahte/ortak-trend korelasyonu yaratabiliyor, detrended bunu giderir
    usdtry_trend = usdtry_ret.rolling(TREND_WINDOW).mean()
    macro_feat = pd.DataFrame({
        "usdtry_ret_1_detrended": usdtry_ret - usdtry_trend,
        "usdtry_vol_10": usdtry_ret.rolling(10).std(),
    }).dropna()

    # asof-merge: her bar icin en son bilinen (gelecegi sizdirmayan) USD/TRY gozlemi
    left = pd.DataFrame(index=target_index).reset_index(names="ts")
    right = macro_feat.reset_index(names="ts")
    merged = pd.merge_asof(
        left.sort_values("ts"), right.sort_values("ts"), on="ts", direction="backward",
    )
    return merged.set_index("ts").reindex(target_index)


def add_relative_strength(features: dict[str, pd.DataFrame]) -> None:
    all_ret1 = pd.DataFrame({t: f["ret_1"] for t, f in features.items()})
    market_ret = all_ret1.mean(axis=1)
    for ticker, f in features.items():
        f["rel_strength"] = f["ret_1"] - market_ret


def add_relative_target(features: dict[str, pd.DataFrame]) -> None:
    # piyasa-geneli enflasyon/devaluasyon trendi tum hisselerde ortak oldugu icin
    # ham getiriyi hedef almak sahte bir "hep yukari" sinyaline yol acar; BIST30
    # ortalamasina gore bagil (excess) ileri getiri bu ortak trendi giderir
    all_fwd = pd.DataFrame({t: f["target_fwd_ret"] for t, f in features.items()})
    market_fwd = all_fwd.mean(axis=1)
    for ticker, f in features.items():
        f["target_fwd_rel_ret"] = f["target_fwd_ret"] - market_fwd

    all_fwd5 = pd.DataFrame({t: f["target_fwd5_ret"] for t, f in features.items()})
    market_fwd5 = all_fwd5.mean(axis=1)
    for ticker, f in features.items():
        f["target_fwd5_rel_ret"] = f["target_fwd5_ret"] - market_fwd5


def run(timeframe: str):
    out_dir = OUT_DIR / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {}
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        df = load_symbol(timeframe, ticker)
        if df is not None:
            raw[ticker] = df

    features = {ticker: build_features(df, timeframe) for ticker, df in raw.items()}
    add_relative_strength(features)
    add_relative_target(features)

    for ticker, f in features.items():
        macro = load_macro_aligned(timeframe, f.index)
        f[["usdtry_ret_1_detrended", "usdtry_vol_10"]] = macro[["usdtry_ret_1_detrended", "usdtry_vol_10"]]
        f = f.dropna()
        f.to_csv(out_dir / f"{ticker}.csv")
        print(f"[{timeframe}] {ticker}: {len(f)} satir, {f.shape[1]} ozellik")


if __name__ == "__main__":
    for tf in ["daily", "hourly"]:
        run(tf)
