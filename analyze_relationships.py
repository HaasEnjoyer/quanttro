from pathlib import Path

import numpy as np
import pandas as pd

from symbols import BIST30

DATA_DIR = Path(__file__).parent / "data" / "hourly"
MIN_OVERLAP = 500  # min ortak bar sayisi, guvenilir korelasyon icin


def load_returns_and_volume():
    returns = {}
    volume_chg = {}
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        returns[ticker] = df["Close"].pct_change()
        volume_chg[ticker] = df["Volume"].replace(0, np.nan).pct_change()
    return pd.DataFrame(returns), pd.DataFrame(volume_chg)


def lead_lag_correlations(signal: pd.DataFrame, target: pd.DataFrame):
    rows = []
    for a in signal.columns:
        a_now = signal[a]
        for b in target.columns:
            if a == b:
                continue
            b_next = target[b].shift(-1)
            paired = pd.concat([a_now, b_next], axis=1).dropna()
            if len(paired) < MIN_OVERLAP:
                continue
            corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            rows.append({"bugun": a, "ertesi_bar": b, "corr": corr, "n": len(paired)})
    return pd.DataFrame(rows).sort_values("corr")


def report(result: pd.DataFrame, title: str, out_name: str):
    print(f"\n########## {title} ##########")
    print(f"Toplam {len(result)} cift hesaplandi (ortak bar >= {MIN_OVERLAP})\n")

    print("=== EN GUCLU TERS ILISKI (negatif korelasyon) ===")
    print(result.head(15).to_string(index=False))

    print("\n=== EN GUCLU DOGRUSAL ILISKI (pozitif korelasyon) ===")
    print(result.tail(15).sort_values("corr", ascending=False).to_string(index=False))

    out_path = Path(__file__).parent / out_name
    result.to_csv(out_path, index=False)
    print(f"\nTum sonuclar {out_path.name} dosyasina kaydedildi.")


if __name__ == "__main__":
    returns, volume_chg = load_returns_and_volume()

    return_result = lead_lag_correlations(returns, returns)
    report(return_result, "RETURN(bugun) -> RETURN(ertesi bar)", "lead_lag_correlations.csv")

    volume_result = lead_lag_correlations(volume_chg, returns)
    report(volume_result, "VOLUME DEGISIMI(bugun) -> RETURN(ertesi bar)", "lead_lag_volume_correlations.csv")
