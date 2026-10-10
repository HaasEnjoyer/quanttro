from pathlib import Path

import yfinance as yf

from fetch_data import TIMEFRAMES

DATA_DIR = Path(__file__).parent / "data" / "macro"

MACRO_TICKERS = {
    "USDTRY": "TRY=X",
}


def fetch_macro():
    for timeframe, cfg in TIMEFRAMES.items():
        out_dir = DATA_DIR / timeframe
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, ticker in MACRO_TICKERS.items():
            df = yf.download(
                ticker,
                interval=cfg["interval"],
                period=cfg["period"],
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print(f"[{timeframe}] {name}: veri yok")
                continue
            df.columns = df.columns.get_level_values(0) if df.columns.nlevels > 1 else df.columns
            if timeframe != "daily":
                df.index = df.index.tz_convert("Europe/Istanbul")
            out_path = out_dir / f"{name}.csv"
            df.to_csv(out_path)
            print(f"[{timeframe}] {name}: {len(df)} satir -> {out_path.name}")


if __name__ == "__main__":
    fetch_macro()
