from pathlib import Path

import pandas as pd
from scipy import stats

from symbols import BIST30
from train_model import load_pooled, walk_forward_folds

N_FOLDS = 5
TARGET_DOWS = {0: "Pazartesi", 1: "Sali"}  # in-sample analizde anlamli cikan iki gun
PRED_DIR = Path(__file__).parent / "data" / "predictions"


def label_tertile(train_vol: pd.Series, test_vol: pd.Series) -> pd.Series:
    low_q, high_q = train_vol.quantile([1 / 3, 2 / 3])
    return pd.cut(test_vol, bins=[-float("inf"), low_q, high_q, float("inf")],
                   labels=["dusuk", "orta", "yuksek"])


def run():
    pooled = load_pooled("daily")
    rows = []

    for train_cutoff, test_start, test_end in walk_forward_folds(pooled.index, N_FOLDS):
        train = pooled[pooled.index <= train_cutoff]
        test = pooled[(pooled.index >= test_start) & (pooled.index <= test_end)].copy()

        # tertile esikleri SADECE train'den, hisse bazinda (gelecek sizdirmasin)
        thresholds = train.groupby("symbol")["volume_zscore"].quantile([1 / 3, 2 / 3]).unstack()
        thresholds.columns = ["low_q", "high_q"]

        test = test.join(thresholds, on="symbol")
        test["vol_tertile"] = "orta"
        test.loc[test["volume_zscore"] <= test["low_q"], "vol_tertile"] = "dusuk"
        test.loc[test["volume_zscore"] >= test["high_q"], "vol_tertile"] = "yuksek"

        for dow, name in TARGET_DOWS.items():
            sub = test[test["day_of_week"] == dow]
            low = sub[sub["vol_tertile"] == "dusuk"]["target_fwd_ret"].dropna()
            high = sub[sub["vol_tertile"] == "yuksek"]["target_fwd_ret"].dropna()
            if len(low) < 20 or len(high) < 20:
                continue
            t, p = stats.ttest_ind(high, low, equal_var=False)
            rows.append({
                "test_start": test_start.date(), "test_end": test_end.date(),
                "gun": name, "n_dusuk": len(low), "n_yuksek": len(high),
                "ort_dusuk": low.mean(), "ort_yuksek": high.mean(),
                "fark": high.mean() - low.mean(), "t": t, "p": p,
            })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n=== Gune gore ozet (5 fold ortalamasi) ===")
    summary = results.groupby("gun").agg(
        ortalama_fark=("fark", "mean"),
        pozitif_fold_orani=("fark", lambda x: (x > 0).mean()),
        ortalama_p=("p", "mean"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PRED_DIR / "volume_dow_rule_validation.csv", index=False)
    return results


if __name__ == "__main__":
    run()
