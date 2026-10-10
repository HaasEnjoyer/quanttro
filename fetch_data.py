import time
from pathlib import Path

import yfinance as yf

from symbols import BIST30

DATA_DIR = Path(__file__).parent / "data"

TIMEFRAMES = {
    "daily": {"interval": "1d", "period": "max"},
    "hourly": {"interval": "60m", "period": "730d"},
    "5min": {"interval": "5m", "period": "60d"},
}


def fetch_symbol(symbol: str, interval: str, period: str):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0) if df.columns.nlevels > 1 else df.columns
    return df


def fetch_all(timeframe: str):
    cfg = TIMEFRAMES[timeframe]
    out_dir = DATA_DIR / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in BIST30:
        df = fetch_symbol(symbol, cfg["interval"], cfg["period"])
        if df is None:
            print(f"[{timeframe}] {symbol}: veri yok, atlandi")
            continue
        out_path = out_dir / f"{symbol.replace('.IS', '')}.csv"
        df.to_csv(out_path)
        print(f"[{timeframe}] {symbol}: {len(df)} satir -> {out_path.name}")
        time.sleep(0.3)


if __name__ == "__main__":
    for tf in TIMEFRAMES:
        fetch_all(tf)
