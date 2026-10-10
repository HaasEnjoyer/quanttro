from pathlib import Path

import pandas as pd
from scipy import stats

from symbols import BIST30

FEATURES_DIR = Path(__file__).parent / "data" / "features" / "daily"
DOW_NAMES = ["Pazartesi", "Sali", "Carsamba", "Persembe", "Cuma"]


def load_pooled() -> pd.DataFrame:
    frames = []
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        path = FEATURES_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df["symbol"] = ticker
        df["day_of_month"] = df.index.day
        frames.append(df)
    return pd.concat(frames)


def by_day_of_week(pooled: pd.DataFrame):
    print("\n=== HAFTANIN GUNUNE GORE: hacim ve getiri ===")
    g = pooled.groupby("day_of_week").agg(
        ortalama_hacim_z=("volume_zscore", "mean"),
        ortalama_getiri=("ret_1", "mean"),
        ortalama_ertesi_getiri=("target_fwd_ret", "mean"),
        n=("ret_1", "size"),
    )
    g.index = [DOW_NAMES[i] for i in g.index]
    print(g.to_string(float_format=lambda x: f"{x:.5f}"))

    # ANOVA: gun-of-week gruplari arasinda hacimde anlamli fark var mi?
    groups = [grp["volume_zscore"].dropna() for _, grp in pooled.groupby("day_of_week")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nHacim icin ANOVA (gunler arasi fark): F={f_stat:.2f}, p-value={p_val:.4g}")

    groups_ret = [grp["target_fwd_ret"].dropna() for _, grp in pooled.groupby("day_of_week")]
    f_stat_r, p_val_r = stats.f_oneway(*groups_ret)
    print(f"Ertesi-gun getiri icin ANOVA: F={f_stat_r:.2f}, p-value={p_val_r:.4g}")


def by_day_of_month(pooled: pd.DataFrame):
    print("\n=== AYIN GUNUNE GORE: hacim ve getiri (ilk/son 5 gun vurgulu) ===")
    g = pooled.groupby("day_of_month").agg(
        ortalama_hacim_z=("volume_zscore", "mean"),
        ortalama_getiri=("ret_1", "mean"),
        ortalama_ertesi_getiri=("target_fwd_ret", "mean"),
        n=("ret_1", "size"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.5f}"))

    groups = [grp["volume_zscore"].dropna() for _, grp in pooled.groupby("day_of_month") if len(grp) > 30]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nHacim icin ANOVA (ayin gunleri arasi fark): F={f_stat:.2f}, p-value={p_val:.4g}")


def volume_x_dayofweek_interaction(pooled: pd.DataFrame):
    print("\n=== ETKILESIM: haftanin gunu x hacim tertile -> ertesi gun getirisi ===")
    pooled = pooled.copy()
    pooled["vol_tertile"] = pooled.groupby("symbol")["volume_zscore"].transform(
        lambda x: pd.qcut(x, 3, labels=["dusuk", "orta", "yuksek"], duplicates="drop")
    )
    pivot = pooled.pivot_table(
        index="day_of_week", columns="vol_tertile", values="target_fwd_ret",
        aggfunc="mean", observed=True,
    )
    pivot.index = [DOW_NAMES[i] for i in pivot.index]
    print(pivot.to_string(float_format=lambda x: f"{x:.5f}"))

    n_pivot = pooled.pivot_table(
        index="day_of_week", columns="vol_tertile", values="target_fwd_ret",
        aggfunc="size", observed=True,
    )
    n_pivot.index = [DOW_NAMES[i] for i in n_pivot.index]
    print("\nGozlem sayilari:")
    print(n_pivot.to_string())


if __name__ == "__main__":
    pooled = load_pooled()
    print(f"Toplam satir: {len(pooled)}, hisse: {pooled['symbol'].nunique()}")
    by_day_of_week(pooled)
    by_day_of_month(pooled)
    volume_x_dayofweek_interaction(pooled)
