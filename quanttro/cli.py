"""quanttro komut satiri arayuzu.

Ornekler:
    quanttro fetch --symbols BIST30 --interval 1d --period max
    quanttro fetch --symbols BIST100 --interval 60m --period 730d --out-dir data/hourly
    quanttro fetch --macro --interval 1d --period max
    quanttro fetch --tickers PAXG-USD BTC-USD --interval 1d --period max
    quanttro validate --file data/daily/THYAO.csv
    quanttro validate --file data/daily/THYAO.csv --jump-threshold 0.3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data.cache import CacheConfig
from .data.fetch import fetch_macro, fetch_ohlcv, save_ohlcv
from .data.symbols import resolve_symbols
from .data.validate import validation_report


def _cmd_fetch(args: argparse.Namespace) -> int:
    cache_config = CacheConfig(ttl_seconds=args.cache_ttl, enabled=not args.no_cache)

    if args.macro:
        data = fetch_macro(interval=args.interval, period=args.period, cache_config=cache_config)
    elif args.tickers:
        data = fetch_ohlcv(args.tickers, interval=args.interval, period=args.period, cache_config=cache_config)
    else:
        symbols = resolve_symbols(args.symbols)
        data = fetch_ohlcv(symbols, interval=args.interval, period=args.period, cache_config=cache_config)

    if not data:
        print("Hic veri cekilemedi.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / args.interval
    written = save_ohlcv(data, out_dir)
    print(f"{len(written)} dosya '{out_dir}' dizinine yazildi ({len(data)} sembol).")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"Dosya bulunamadi: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    report = validation_report(
        df,
        jump_threshold=args.jump_threshold,
        zero_volume_min_run=args.zero_volume_min_run,
        stale_min_run=args.stale_min_run,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Dosya: {path} ({report['n_rows']} satir)")
        print(
            f"  Fiyat sicramasi (>= %{args.jump_threshold * 100:.0f}): "
            f"{report['price_jumps']['count']} satir"
        )
        print(
            f"  Sifir-hacim run'lari: {report['zero_volume_runs']['count']} satir / "
            f"{report['zero_volume_runs']['n_runs']} run"
        )
        print(
            f"  Donmus fiyat run'lari: {report['stale_price_runs']['count']} satir / "
            f"{report['stale_price_runs']['n_runs']} run"
        )
        print(f"  Temiz mi: {report['is_clean']}")

    return 0 if report["is_clean"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quanttro", description="Kantitatif arastirma veri araclari (BIST dahil, herhangi bir yfinance ticker'i icin)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="OHLCV / makro / genel ticker verisi cek")
    fetch_p.add_argument(
        "--symbols", default="BIST30",
        help="Hazir sembol seti adi (BIST30 / BIST100), varsayilan BIST30",
    )
    fetch_p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Ozel yfinance ticker listesi (ornek: PAXG-USD BTC-USD THYAO.IS)",
    )
    fetch_p.add_argument("--macro", action="store_true", help="Makro gostergeleri cek (varsayilan USDTRY=X)")
    fetch_p.add_argument("--interval", default="1d", help="yfinance interval (1d, 60m, 5m, ...)")
    fetch_p.add_argument("--period", default="max", help="yfinance period (max, 730d, 60d, ...)")
    fetch_p.add_argument("--out-dir", default=None, help="Cikis dizini (varsayilan: data/<interval>)")
    fetch_p.add_argument("--no-cache", action="store_true", help="Cache katmanini devre disi birak")
    fetch_p.add_argument("--cache-ttl", type=float, default=24 * 60 * 60, help="Cache TTL (saniye)")
    fetch_p.set_defaults(func=_cmd_fetch)

    validate_p = sub.add_parser("validate", help="Bir OHLCV CSV dosyasinin veri kalitesini kontrol et")
    validate_p.add_argument("--file", required=True, help="Kontrol edilecek CSV dosyasi")
    validate_p.add_argument("--jump-threshold", type=float, default=0.5, help="Fiyat sicramasi esigi (0-1)")
    validate_p.add_argument("--zero-volume-min-run", type=int, default=3, help="Sifir-hacim run minimum uzunlugu")
    validate_p.add_argument("--stale-min-run", type=int, default=5, help="Donmus fiyat run minimum uzunlugu")
    validate_p.add_argument("--json", action="store_true", help="Raporu JSON olarak yazdir")
    validate_p.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
