from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from symbols import BIST30

DATA_DIR = Path(__file__).parent / "data" / "daily"
RETURN_CLIP = 0.5  # haftalik getiri icin daha gevsek bir winsorize


def build_weekly(ticker: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    weekly_close = df["Close"].resample("W-FRI").last()
    weekly_volume = df["Volume"].resample("W-FRI").sum()
    weekly_ret = weekly_close.pct_change().clip(-RETURN_CLIP, RETURN_CLIP)

    out = pd.DataFrame({
        "close": weekly_close,
        "volume": weekly_volume,
        "ret": weekly_ret,
    }).dropna()

    # haftalik hacmin kendi tarihsel dagilimina gore z-skoru (hisse bazinda)
    out["volume_z"] = (out["volume"] - out["volume"].rolling(26).mean()) / out["volume"].rolling(26).std()
    # yonlu hacim: o hafta yukari kapandiysa +hacim, asagi kapandiysa -hacim
    out["signed_volume_z"] = out["volume_z"] * np.sign(out["ret"])
    out["direction"] = np.where(out["ret"] >= 0, "yukari", "asagi")
    out["target_next_week_ret"] = out["ret"].shift(-1)
    return out.dropna()


def load_pooled() -> pd.DataFrame:
    frames = []
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        weekly = build_weekly(ticker)
        if weekly is not None:
            weekly["symbol"] = ticker
            frames.append(weekly)
    return pd.concat(frames)


def cross_tab_analysis(pooled: pd.DataFrame):
    pooled = pooled.copy()
    pooled["vol_tertile"] = pooled.groupby("symbol")["volume_z"].transform(
        lambda x: pd.qcut(x, 3, labels=["dusuk", "orta", "yuksek"], duplicates="drop")
    )

    print("=== BU HAFTA (yon x hacim tertile) -> GELECEK HAFTA ortalama getiri ===")
    pivot_mean = pooled.pivot_table(
        index="direction", columns="vol_tertile", values="target_next_week_ret",
        aggfunc="mean", observed=True,
    )
    print(pivot_mean.to_string(float_format=lambda x: f"{x:.5f}"))

    print("\n=== Gozlem sayilari ===")
    pivot_n = pooled.pivot_table(
        index="direction", columns="vol_tertile", values="target_next_week_ret",
        aggfunc="size", observed=True,
    )
    print(pivot_n.to_string())

    print("\n=== GELECEK HAFTA pozitif getiri orani (basari orani) ===")
    pivot_winrate = pooled.pivot_table(
        index="direction", columns="vol_tertile",
        values="target_next_week_ret", aggfunc=lambda x: (x > 0).mean(), observed=True,
    )
    print(pivot_winrate.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Istatistiksel anlamlilik: 'yukari+yuksek hacim' vs 'yukari+dusuk hacim' ===")
    up_high = pooled[(pooled["direction"] == "yukari") & (pooled["vol_tertile"] == "yuksek")]["target_next_week_ret"].dropna()
    up_low = pooled[(pooled["direction"] == "yukari") & (pooled["vol_tertile"] == "dusuk")]["target_next_week_ret"].dropna()
    t, p = stats.ttest_ind(up_high, up_low, equal_var=False)
    print(f"yukari+yuksek (n={len(up_high)}, ort={up_high.mean():.5f}) vs yukari+dusuk (n={len(up_low)}, ort={up_low.mean():.5f})")
    print(f"t={t:.3f}, p={p:.4g}")

    print("\n=== Istatistiksel anlamlilik: 'asagi+yuksek hacim' vs 'asagi+dusuk hacim' ===")
    down_high = pooled[(pooled["direction"] == "asagi") & (pooled["vol_tertile"] == "yuksek")]["target_next_week_ret"].dropna()
    down_low = pooled[(pooled["direction"] == "asagi") & (pooled["vol_tertile"] == "dusuk")]["target_next_week_ret"].dropna()
    t2, p2 = stats.ttest_ind(down_high, down_low, equal_var=False)
    print(f"asagi+yuksek (n={len(down_high)}, ort={down_high.mean():.5f}) vs asagi+dusuk (n={len(down_low)}, ort={down_low.mean():.5f})")
    print(f"t={t2:.3f}, p={p2:.4g}")

    print("\n=== Surekli (continuous) iliski: signed_volume_z -> gelecek hafta getirisi ===")
    corr = pooled["signed_volume_z"].corr(pooled["target_next_week_ret"], method="spearman")
    print(f"Spearman korelasyon: {corr:.4f}  (n={len(pooled)})")


def walk_forward_regime_check(pooled: pd.DataFrame) -> pd.DataFrame:
    from train_model import walk_forward_folds

    rows = []
    for train_cutoff, test_start, test_end in walk_forward_folds(pooled.index, 5):
        train = pooled[pooled.index <= train_cutoff]
        test = pooled[(pooled.index >= test_start) & (pooled.index <= test_end)].copy()

        thresholds = train.groupby("symbol")["volume_z"].quantile([1 / 3, 2 / 3]).unstack()
        thresholds.columns = ["low_q", "high_q"]
        test = test.join(thresholds, on="symbol")
        test["vol_tertile"] = "orta"
        test.loc[test["volume_z"] <= test["low_q"], "vol_tertile"] = "dusuk"
        test.loc[test["volume_z"] >= test["high_q"], "vol_tertile"] = "yuksek"

        up = test[test["direction"] == "yukari"]
        up_high = up[up["vol_tertile"] == "yuksek"]["target_next_week_ret"].dropna()
        up_low = up[up["vol_tertile"] == "dusuk"]["target_next_week_ret"].dropna()

        low_all = test[test["vol_tertile"] == "dusuk"]["target_next_week_ret"].dropna()
        high_all = test[test["vol_tertile"] == "yuksek"]["target_next_week_ret"].dropna()

        if min(len(up_high), len(up_low), len(low_all), len(high_all)) < 20:
            continue

        rows.append({
            "fold_start": test_start.date(), "fold_end": test_end.date(),
            "momentum_fark": up_high.mean() - up_low.mean(),  # yukari+yuksek - yukari+dusuk
            "genel_fark": low_all.mean() - high_all.mean(),   # dusuk - yuksek (genel)
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pooled = load_pooled()
    print(f"Toplam hafta-hisse gozlemi: {len(pooled)}, hisse: {pooled['symbol'].nunique()}\n")
    cross_tab_analysis(pooled)

    regime_df = walk_forward_regime_check(pooled)
    out_path = Path(__file__).parent / "data" / "predictions" / "weekly_volume_regime.csv"
    regime_df.to_csv(out_path, index=False)
    print(f"\nRejim kirilma verisi kaydedildi: {out_path}")
