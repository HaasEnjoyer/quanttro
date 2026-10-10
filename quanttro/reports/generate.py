"""Bir sembol icin otomatik ozet rapor (Markdown ya da HTML) ureten fonksiyonlar.

Amac, her hisse icin elle yazilan analiz script'lerini (analyze_autocorrelation.py,
analyze_volume_calendar.py vb.) tekrarlamadan, fiyat/hacim ozeti + anomali tespiti
(quanttro.diagnostics.anomalies) iceren kisa, okunabilir bir rapor dosyasi uretmek.
Karmasik bir rapor motoru degil - tek bir hisseyi hizlica gozden gecirmek icin
dusunulmus, basit bir string-sablon yaklasimidir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quanttro.diagnostics.anomalies import (
    LIMIT_THRESHOLD,
    detect_limit_streaks,
    limit_frequency_by_year,
)

TRADING_DAYS_PER_YEAR = 252


def _extract_close_volume_returns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """df hem ham OHLCV (buyuk harfli "Close"/"Volume", fetch_data.py formati) hem de
    ozellik tablosu (kucuk harfli "close"/"ret_1"/"volume_zscore", feature_engineering.py
    formati) olarak gelebilir; her ikisini de destekler."""
    if "Close" in df.columns:
        close = df["Close"]
    elif "close" in df.columns:
        close = df["close"]
    else:
        raise ValueError("df bir 'Close' ya da 'close' kolonu icermeli.")

    if "Volume" in df.columns:
        volume = df["Volume"]
    elif "volume" in df.columns:
        volume = df["volume"]
    else:
        volume = pd.Series(np.nan, index=df.index)

    if "ret_1" in df.columns:
        returns = df["ret_1"]
    else:
        returns = close.pct_change()

    return close, volume, returns


def _summarize(symbol: str, df: pd.DataFrame) -> dict:
    close, volume, returns = _extract_close_volume_returns(df)
    returns = returns.dropna()

    price_stats = {
        "start": df.index.min(),
        "end": df.index.max(),
        "n_bar": len(df),
        "last_close": float(close.iloc[-1]) if len(close) else float("nan"),
        "period_return": float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else float("nan"),
        "mean_daily_return": float(returns.mean()) if len(returns) else float("nan"),
        "std_daily_return": float(returns.std()) if len(returns) else float("nan"),
        "approx_annual_vol": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) else float("nan"),
    }

    has_volume = volume.notna().any()
    volume_stats = {
        "mean_volume": float(volume.mean()) if has_volume else float("nan"),
        "last_20_mean_volume": float(volume.tail(20).mean()) if has_volume else float("nan"),
    }

    streaks = detect_limit_streaks(returns, threshold=LIMIT_THRESHOLD, min_streak=2)
    yearly = limit_frequency_by_year(returns, threshold=LIMIT_THRESHOLD)

    return {
        "symbol": symbol,
        "price_stats": price_stats,
        "volume_stats": volume_stats,
        "streaks": streaks,
        "yearly": yearly,
    }


def _fmt_date(value) -> str:
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _render_markdown(summary: dict) -> str:
    symbol = summary["symbol"]
    p, v = summary["price_stats"], summary["volume_stats"]
    streaks, yearly = summary["streaks"], summary["yearly"]

    lines = [
        f"# {symbol} - Ozet ve Anomali Raporu",
        "",
        "*Bu rapor `quanttro.reports.generate.generate_symbol_report` tarafindan otomatik uretildi.*",
        "",
        "## Fiyat Ozeti",
        "",
        f"- Donem: {_fmt_date(p['start'])} -> {_fmt_date(p['end'])} ({p['n_bar']} bar)",
        f"- Son kapanis: {p['last_close']:.2f}",
        f"- Donem getirisi: {p['period_return']:.2%}",
        f"- Ortalama gunluk getiri: {p['mean_daily_return']:.4%}",
        f"- Gunluk getiri std: {p['std_daily_return']:.4%}",
        f"- Yaklasik yillik volatilite (std * sqrt(252)): {p['approx_annual_vol']:.2%}",
        "",
        "## Hacim Ozeti",
        "",
        f"- Ortalama hacim: {v['mean_volume']:,.0f}",
        f"- Son 20 bar ortalama hacim: {v['last_20_mean_volume']:,.0f}",
        "",
        "## Tavan/Taban Serisi Tespiti",
        "",
        "(`detect_limit_streaks`, esik = %9.5, min. 2 ardisik gun)",
        "",
    ]

    if streaks.empty:
        lines.append("Ardisik tavan/taban serisi bulunamadi.")
    else:
        longest = int(streaks["length"].max())
        lines.append(f"Toplam **{len(streaks)}** seri bulundu (en uzunu: **{longest} gun**).")
        lines.append("")
        lines.append("| Baslangic | Bitis | Uzunluk (gun) | Yon |")
        lines.append("|---|---|---|---|")
        for _, row in streaks.head(10).iterrows():
            lines.append(
                f"| {_fmt_date(row['start_date'])} | {_fmt_date(row['end_date'])} | "
                f"{row['length']} | {row['direction']} |"
            )
        if len(streaks) > 10:
            lines.append(f"\n*(ilk 10 / {len(streaks)} seri gosteriliyor, uzunluga gore siralanmis)*")

    lines += ["", "## Yillik Tavan/Taban Orani", "", "(`limit_frequency_by_year`)", ""]

    if yearly.empty:
        lines.append("Yeterli veri yok.")
    else:
        lines.append("| Yil | Gun Sayisi | Tavan Orani | Taban Orani | Toplam |")
        lines.append("|---|---|---|---|---|")
        for _, row in yearly.iterrows():
            lines.append(
                f"| {int(row['year'])} | {int(row['n_gun'])} | {row['tavan_orani']:.2%} | "
                f"{row['taban_orani']:.2%} | {row['toplam_limit_orani']:.2%} |"
            )

    lines += [
        "",
        "---",
        "_Not: Yuksek tavan/taban orani veya uzun ardisik seriler tek basina "
        "\"ongorulebilirlik\" anlamina gelmez; dusuk-float/manipulasyon/distress "
        "sinyali olabilir. Bkz. `docs/KNOWN_PITFALLS.md`._",
    ]
    return "\n".join(lines)


def _render_html(summary: dict) -> str:
    symbol = summary["symbol"]
    p, v = summary["price_stats"], summary["volume_stats"]
    streaks, yearly = summary["streaks"], summary["yearly"]

    def streak_rows() -> str:
        if streaks.empty:
            return "<tr><td colspan='4'>Ardisik tavan/taban serisi bulunamadi.</td></tr>"
        rows = []
        for _, row in streaks.head(10).iterrows():
            rows.append(
                f"<tr><td>{_fmt_date(row['start_date'])}</td><td>{_fmt_date(row['end_date'])}</td>"
                f"<td>{row['length']}</td><td>{row['direction']}</td></tr>"
            )
        return "\n".join(rows)

    def yearly_rows() -> str:
        if yearly.empty:
            return "<tr><td colspan='5'>Yeterli veri yok.</td></tr>"
        rows = []
        for _, row in yearly.iterrows():
            rows.append(
                f"<tr><td>{int(row['year'])}</td><td>{int(row['n_gun'])}</td>"
                f"<td>{row['tavan_orani']:.2%}</td><td>{row['taban_orani']:.2%}</td>"
                f"<td>{row['toplam_limit_orani']:.2%}</td></tr>"
            )
        return "\n".join(rows)

    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>{symbol} - Anomali Raporu</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 860px; margin: 2rem auto; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f2f2f2; }}
  .note {{ color: #666; font-size: .9rem; margin-top: 2rem; border-top: 1px solid #ddd; padding-top: .8rem; }}
</style>
</head>
<body>
<h1>{symbol} - Ozet ve Anomali Raporu</h1>
<p><em>Bu rapor quanttro.reports.generate.generate_symbol_report tarafindan otomatik uretildi.</em></p>

<h2>Fiyat Ozeti</h2>
<ul>
  <li>Donem: {_fmt_date(p['start'])} -&gt; {_fmt_date(p['end'])} ({p['n_bar']} bar)</li>
  <li>Son kapanis: {p['last_close']:.2f}</li>
  <li>Donem getirisi: {p['period_return']:.2%}</li>
  <li>Ortalama gunluk getiri: {p['mean_daily_return']:.4%}</li>
  <li>Gunluk getiri std: {p['std_daily_return']:.4%}</li>
  <li>Yaklasik yillik volatilite: {p['approx_annual_vol']:.2%}</li>
</ul>

<h2>Hacim Ozeti</h2>
<ul>
  <li>Ortalama hacim: {v['mean_volume']:,.0f}</li>
  <li>Son 20 bar ortalama hacim: {v['last_20_mean_volume']:,.0f}</li>
</ul>

<h2>Tavan/Taban Serisi Tespiti</h2>
<p>(esik = %9.5, min. 2 ardisik gun)</p>
<table>
  <tr><th>Baslangic</th><th>Bitis</th><th>Uzunluk (gun)</th><th>Yon</th></tr>
  {streak_rows()}
</table>

<h2>Yillik Tavan/Taban Orani</h2>
<table>
  <tr><th>Yil</th><th>Gun Sayisi</th><th>Tavan Orani</th><th>Taban Orani</th><th>Toplam</th></tr>
  {yearly_rows()}
</table>

<p class="note">Not: Yuksek tavan/taban orani veya uzun ardisik seriler tek basina
"ongorulebilirlik" anlamina gelmez; dusuk-float/manipulasyon/distress sinyali olabilir.
Bkz. docs/KNOWN_PITFALLS.md.</p>
</body>
</html>
"""


def generate_symbol_report(
    symbol: str,
    df: pd.DataFrame,
    output_path: str,
    format: str = "markdown",
) -> None:
    """`symbol` icin fiyat/hacim ozeti + anomali tespiti iceren bir rapor dosyasi uretir.

    Parametreler
    ----------
    symbol : str
        Rapor basliginda gosterilecek sembol adi (ornek: "KONTR", "THYAO").
    df : pd.DataFrame
        DatetimeIndex'li fiyat serisi. Hem ham OHLCV ("Close", "Volume" - fetch_data.py
        ciktisi) hem de ozellik tablosu ("close", "ret_1", ... - feature_engineering.py
        ciktisi) formatini destekler.
    output_path : str
        Rapor dosyasinin yazilacagi yol. Ust dizinler yoksa otomatik olusturulur.
    format : str
        "markdown" (varsayilan, .md icerik) ya da "html" (.html icerik).

    Donus
    -----
    None - sonuc dogrudan `output_path`'e yazilir.

    Ornek
    -----
    >>> df = pd.read_csv("data/daily/KONTR.csv", index_col=0, parse_dates=True)
    >>> generate_symbol_report("KONTR", df, "reports_out/KONTR.md")
    """
    summary = _summarize(symbol, df)

    if format == "markdown":
        content = _render_markdown(summary)
    elif format == "html":
        content = _render_html(summary)
    else:
        raise ValueError(f"Desteklenmeyen format: {format!r} ('markdown' ya da 'html' olmali)")

    disclaimer = (
        "\n\n---\n*Bu rapor quanttro tarafindan otomatik uretilmistir. "
        "Yatirim tavsiyesi degildir; gecmis performans gelecegi garanti etmez.*\n"
        if format == "markdown"
        else '\n<hr><p style="font-size:12px;color:#888;">'
        "This report was generated automatically by quanttro. "
        "Not investment advice; past performance does not guarantee future results.</p>\n"
    )
    content += disclaimer

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
