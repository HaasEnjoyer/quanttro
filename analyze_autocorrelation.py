from pathlib import Path

import numpy as np
import pandas as pd

from symbols import BIST30

FEATURES_DIR = Path(__file__).parent / "data" / "features" / "daily"
LAGS = [1, 2, 3, 4, 5, 10, 20]


def acf_with_significance(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    series = series.dropna()
    n = len(series)
    se = 1 / np.sqrt(n)  # beyaz gurultu altinda ACF'nin yaklasik standart hatasi
    rows = []
    for lag in lags:
        ac = series.autocorr(lag)
        z = ac / se
        rows.append({"lag": lag, "acf": ac, "z": z, "anlamli_95": abs(z) > 1.96})
    return pd.DataFrame(rows)


def per_symbol_summary() -> pd.DataFrame:
    rows = []
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        path = FEATURES_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        ret_acf = acf_with_significance(df["ret_1"], LAGS)
        vol_acf = acf_with_significance(df["volume_zscore"], LAGS)

        row = {"symbol": ticker, "n_bar": len(df)}
        for lag in LAGS:
            row[f"ret_acf_lag{lag}"] = ret_acf.loc[ret_acf["lag"] == lag, "acf"].iloc[0]
            row[f"vol_acf_lag{lag}"] = vol_acf.loc[vol_acf["lag"] == lag, "acf"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary = per_symbol_summary()

    print("=== Getiri otokorelasyonu (lag-1), hisseye gore siralanmis ===")
    print(summary[["symbol", "n_bar", "ret_acf_lag1", "ret_acf_lag5", "ret_acf_lag20"]]
          .sort_values("ret_acf_lag1", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Hacim otokorelasyonu (lag-1), hisseye gore siralanmis ===")
    print(summary[["symbol", "vol_acf_lag1", "vol_acf_lag5", "vol_acf_lag20"]]
          .sort_values("vol_acf_lag1", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # anlamlilik esigi: n~5000-6700 bar icin SE~0.012-0.014, yani |acf|>0.03 kabaca z>2
    print("\n=== Lag-1 getiri otokorelasyonu |acf| > 0.03 olan hisseler (olasi kendine ozgu davranis) ===")
    threshold = 0.03
    flagged = summary[summary["ret_acf_lag1"].abs() > threshold][["symbol", "n_bar", "ret_acf_lag1"]]
    print(flagged.sort_values("ret_acf_lag1", key=abs, ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = Path(__file__).parent / "data" / "predictions" / "autocorrelation_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nTum sonuclar kaydedildi: {out_path}")
