"""QuantStats tarzi, kapsamli bir performans karnesi ("tearsheet") uretici.

Bu modul, `quanttro.backtest.metrics` icinde KANITLANMIS/KESIN mevcut olan
metrik fonksiyonlarini (``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``,
``calmar_ratio``, ``hit_rate``, ``profit_factor``) DOGRUDAN import ederek
kullanir - bu formulleri burada tekrar YAZMAZ.

``metrics.py`` modulu bu proje uzerinde PARALEL olarak genisletilmektedir
(``alpha_beta``, ``omega_ratio``, ``tail_ratio``, ``value_at_risk``,
``conditional_value_at_risk``, ``capture_ratio``, ``kelly_criterion``,
``risk_of_ruin``, ``ulcer_index``, ``common_sense_ratio``, ``avg_win``,
``avg_loss``, ``payoff_ratio``, ``recovery_factor``, ``gain_to_pain_ratio``
gibi fonksiyonlar baska bir gelistirici tarafindan ekleniyor). Bu
fonksiyonlarin bu dosya yazilirken HENUZ mevcut olmama (ya da farkli bir
imzayla gelme) ihtimaline karsi, bu isimler asla dogrudan ``from ... import``
EDILMEZ. Bunun yerine :func:`_safe_metric` / :func:`_safe_metric_with_benchmark`
araciligiyla ``getattr(metrics_module, "...", None)`` ile GUVENLI erisim
yapilir; fonksiyon yoksa ya da cagri herhangi bir sebeple (bilinmeyen imza,
beklenmeyen veri sekli, ic hata) basarisiz olursa, o metrik basitce rapordan
DUSURULUR - rapor asla kirilmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quanttro.backtest import metrics as _metrics
from quanttro.backtest.metrics import (
    calmar_ratio,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)

TRADING_DAYS_PER_YEAR = 252

__all__ = [
    "tearsheet",
    "monthly_returns_table",
    "rolling_sharpe",
    "drawdown_periods",
]

_MONTH_LABELS = ["Oca", "Sub", "Mar", "Nis", "May", "Haz", "Tem", "Agu", "Eyl", "Eki", "Kas", "Ara"]

# metrics.py henuz icermeyebilecegi (paralel gelistirilen) opsiyonel metrikler.
# (fonksiyon adi, rapordaki etiket, bicim: "pct" -> yuzde, "ratio" -> duz sayi)
_OPTIONAL_METRIC_SPECS: list[tuple[str, str, str]] = [
    ("omega_ratio", "Omega Ratio", "ratio"),
    ("tail_ratio", "Tail Ratio", "ratio"),
    ("value_at_risk", "VaR (gunluk, %95)", "pct"),
    ("conditional_value_at_risk", "CVaR (gunluk, %95)", "pct"),
    ("kelly_criterion", "Kelly Kriteri", "ratio"),
    ("risk_of_ruin", "Risk of Ruin", "pct"),
    ("ulcer_index", "Ulcer Index", "pct"),
    ("common_sense_ratio", "Common Sense Ratio", "ratio"),
    ("avg_win", "Ortalama Kazanc (gunluk)", "pct"),
    ("avg_loss", "Ortalama Kayip (gunluk)", "pct"),
    ("payoff_ratio", "Payoff Ratio", "ratio"),
    ("recovery_factor", "Recovery Factor", "ratio"),
    ("gain_to_pain_ratio", "Gain-to-Pain Ratio", "ratio"),
]


# ---------------------------------------------------------------------------
# metrics.py'ye GUVENLI (henuz var olmayabilecek fonksiyonlar icin) erisim
# ---------------------------------------------------------------------------

_UNAVAILABLE = object()


def _try_calls(func, attempts: list[tuple[tuple, dict]]) -> Any:
    """``attempts`` listesindeki (args, kwargs) kombinasyonlarini sirayla
    dener; ilk basarili (exception firlatmayan) cagrinin sonucunu dondurur.
    Hepsi basarisiz olursa :data:`_UNAVAILABLE` sentinel'ini dondurur.

    Boylece hem "fonksiyon mevcut degil" hem de "fonksiyon mevcut ama
    beklenen imzayla/veri sekliyle uyumsuz" durumlari ayni sekilde (metrik
    rapordan dusurulerek) ele alinir.
    """
    for args, kwargs in attempts:
        try:
            return func(*args, **kwargs)
        except Exception:
            continue
    return _UNAVAILABLE


def _safe_metric(
    func_name: str,
    returns: pd.Series,
    cumulative: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Any:
    """``quanttro.backtest.metrics.<func_name>`` fonksiyonunu, henuz
    mevcut olmayabilecegini ve imzasinin tam olarak bilinmedigini goz onune
    alarak guvenli bicimde cagirir. Fonksiyon yoksa ya da denenen hicbir
    cagri sekli basarili olmazsa :data:`_UNAVAILABLE` doner (cagiran kod bu
    durumda ilgili satiri rapora EKLEMEMELIDIR). Fonksiyon basariyla
    calisip NaN donerse (orn. yetersiz veri), bu deger oldugu gibi
    dondurulur - tabloda "N/A" olarak bicimlenir ama satir GOSTERILIR,
    cunku metrik mevcuttur, sadece bu veri icin tanimsizdir.
    """
    func = getattr(_metrics, func_name, None)
    if func is None or not callable(func):
        return _UNAVAILABLE
    attempts: list[tuple[tuple, dict]] = [
        ((returns,), {}),
        ((returns,), {"periods_per_year": periods_per_year}),
    ]
    if cumulative is not None:
        attempts.append(((cumulative,), {}))
    return _try_calls(func, attempts)


def _safe_metric_with_benchmark(
    func_name: str,
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Any:
    """:func:`_safe_metric` ile ayni mantik, ancak benchmark getiri serisi
    de gerektiren metrikler (``alpha_beta``, ``capture_ratio`` vb.) icin."""
    func = getattr(_metrics, func_name, None)
    if func is None or not callable(func):
        return _UNAVAILABLE
    attempts: list[tuple[tuple, dict]] = [
        ((returns, benchmark), {}),
        ((returns, benchmark), {"periods_per_year": periods_per_year}),
    ]
    return _try_calls(func, attempts)


# ---------------------------------------------------------------------------
# Bicimleme yardimcilari
# ---------------------------------------------------------------------------


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value is _UNAVAILABLE:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "N/A"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}%}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or value is _UNAVAILABLE:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "N/A"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def _fmt_date(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def _format_alpha_beta(value: Any) -> str | None:
    """``alpha_beta`` fonksiyonunun donus sekli bilinmedigi icin (tuple,
    list ya da dict olabilir) esnek bicimde ele alinir."""
    try:
        if isinstance(value, dict):
            alpha, beta = value.get("alpha"), value.get("beta")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            alpha, beta = value
        else:
            return None
        return f"alpha={_fmt_pct(alpha, 4)}, beta={_fmt_num(beta, 3)}"
    except Exception:
        return None


def _format_capture(value: Any) -> str | None:
    """``capture_ratio`` fonksiyonunun donus sekli bilinmedigi icin esnek
    bicimde ele alinir (tek sayi, (up, down) tuple'i ya da dict)."""
    try:
        if isinstance(value, dict):
            up, down = value.get("up"), value.get("down")
            return f"up={_fmt_num(up, 3)}, down={_fmt_num(down, 3)}"
        if isinstance(value, (tuple, list)) and len(value) == 2:
            up, down = value
            return f"up={_fmt_num(up, 3)}, down={_fmt_num(down, 3)}"
        return _fmt_num(value, 3)
    except Exception:
        return None


def _clean_returns(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna().astype(float)
    try:
        r.index = pd.to_datetime(r.index)
    except Exception:
        pass
    return r.sort_index()


# ---------------------------------------------------------------------------
# Temel hesaplamalar (metrics.py fonksiyonlarini KULLANIR, yeniden yazmaz)
# ---------------------------------------------------------------------------


def _core_summary(r: pd.Series) -> dict:
    n = len(r)
    cumulative = (1.0 + r).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0) if n else float("nan")

    years = n / TRADING_DAYS_PER_YEAR if n else float("nan")
    if n and years > 0 and cumulative.iloc[-1] > 0:
        cagr = float(cumulative.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else float("nan")

    return {
        "n_obs": n,
        "cumulative": cumulative,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe_ratio(r, periods_per_year=TRADING_DAYS_PER_YEAR),
        "sortino": sortino_ratio(r, periods_per_year=TRADING_DAYS_PER_YEAR),
        "max_dd": max_drawdown(cumulative),
        "calmar": calmar_ratio(r, periods_per_year=TRADING_DAYS_PER_YEAR),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
    }


def _collect_optional_metrics(r: pd.Series, cumulative: pd.Series) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for func_name, label, kind in _OPTIONAL_METRIC_SPECS:
        value = _safe_metric(func_name, r, cumulative=cumulative)
        if value is _UNAVAILABLE:
            continue
        formatted = _fmt_pct(value) if kind == "pct" else _fmt_num(value, 4)
        rows.append((label, formatted))
    return rows


def _collect_benchmark_section(r: pd.Series, b: pd.Series) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    common_idx = r.index.intersection(b.index)
    if len(common_idx) < 2:
        return rows
    r_aligned = r.loc[common_idx]
    b_aligned = b.loc[common_idx]

    try:
        corr = float(r_aligned.corr(b_aligned))
        rows.append(("Korelasyon (getiri)", _fmt_num(corr, 3)))
    except Exception:
        pass

    bench_cum = (1.0 + b_aligned).cumprod()
    if len(bench_cum):
        rows.append(("Benchmark Toplam Getiri", _fmt_pct(float(bench_cum.iloc[-1] - 1.0))))

    alpha_beta_value = _safe_metric_with_benchmark("alpha_beta", r_aligned, b_aligned)
    if alpha_beta_value is not _UNAVAILABLE:
        formatted = _format_alpha_beta(alpha_beta_value)
        if formatted is not None:
            rows.append(("Alpha / Beta", formatted))

    capture_value = _safe_metric_with_benchmark("capture_ratio", r_aligned, b_aligned)
    if capture_value is not _UNAVAILABLE:
        formatted = _format_capture(capture_value)
        if formatted is not None:
            rows.append(("Capture Ratio", formatted))

    return rows


# ---------------------------------------------------------------------------
# Public: monthly_returns_table / rolling_sharpe / drawdown_periods
# ---------------------------------------------------------------------------


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Getiri serisini yil x ay pivot tablosuna donusturur.

    Her hucre, o yil-ay icindeki basit (gunluk) getirilerin BILESIK
    (compound) toplam getirisidir: ``(1 + r).prod() - 1``. Satirlar yil
    (index), sutunlar 1-12 arasi ay numarasi olarak duzenlenir; o ay-yil
    icin veri olmayan hucreler NaN'dir.

    Parametreler
    ----------
    returns : pd.Series
        DatetimeIndex'li basit (aritmetik) getiri serisi.

    Donus
    -----
    pd.DataFrame
        Index = yil (int), kolonlar = 1..12 (ay numarasi).
    """
    r = _clean_returns(returns)
    if r.empty:
        return pd.DataFrame(columns=range(1, 13))

    monthly = r.groupby([r.index.year, r.index.month]).apply(lambda x: float((1.0 + x).prod() - 1.0))
    monthly.index = monthly.index.set_names(["year", "month"])
    table = monthly.unstack("month")
    table = table.reindex(columns=range(1, 13))
    table.index.name = "year"
    return table


def rolling_sharpe(returns: pd.Series, window: int = 252, periods_per_year: int = 252) -> pd.Series:
    """Kayan (rolling) pencerede Sharpe orani serisi.

    Performans icin ``metrics.sharpe_ratio``'yu her pencerede tekrar tekrar
    cagirmak yerine, formul dogrudan rolling mean/std uzerinden hesaplanir::

        rolling_sharpe_t = mean(r[t-window+1:t+1]) / std(r[t-window+1:t+1], ddof=1)
                           * sqrt(periods_per_year)

    Risksiz oran dahil edilmez (0 kabul edilir) - amac, zaman icindeki
    Sharpe DALGALANMASINI gormektir, nokta tahmini icin ``metrics.sharpe_ratio``
    kullanilmalidir.

    Parametreler
    ----------
    returns : pd.Series
        Basit getiri serisi.
    window : int
        Kayan pencere boyutu (periyot sayisi, varsayilan 252).
    periods_per_year : int
        Yillliklandirma icin periyot/yil (varsayilan 252).

    Donus
    -----
    pd.Series
        ``returns`` ile ayni index'e sahip, ilk ``window - 1`` degeri (ve
        pencere ici std'si ~0 olan noktalari) NaN olan Sharpe serisi.
    """
    r = pd.Series(returns).astype(float)
    rolling_mean = r.rolling(window=window).mean()
    rolling_std = r.rolling(window=window).std(ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    result = result.where(rolling_std.abs() > 1e-12, np.nan)
    result.name = "rolling_sharpe"
    return result


def drawdown_periods(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Getiri serisindeki en buyuk ``top_n`` drawdown donemini tablo olarak
    dondurur.

    Her satir bir drawdown donemini tanimlar:

    - ``start``: donemin basladigi tepe (peak) tarihi - running max'in son
      kez guncellendigi, dususun hemen basladigi onceki bar.
    - ``valley``: o donem icindeki en dusuk (dip) noktanin tarihi.
    - ``recovery``: egrinin tekrar onceki tepeyi (``start`` gunundeki
      degeri) astigi ilk tarih; donem rapor sonuna kadar toparlanmadiysa
      ``None``.
    - ``depth``: o donemin en kotu (NEGATIF) yuzdesel dususu.

    Donemler ``depth``'e gore (en kotuden en aza) siralanir.

    Parametreler
    ----------
    returns : pd.Series
        DatetimeIndex'li basit getiri serisi.
    top_n : int
        Dondurulecek en derin drawdown donemi sayisi.

    Donus
    -----
    pd.DataFrame
        Kolonlar: ``start``, ``valley``, ``recovery``, ``depth``.
    """
    columns = ["start", "valley", "recovery", "depth"]
    r = _clean_returns(returns)
    if r.empty:
        return pd.DataFrame(columns=columns)

    cumulative = (1.0 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0

    underwater = drawdown < -1e-12
    if not underwater.any():
        return pd.DataFrame(columns=columns)

    # Her "peak" (dd ~ 0) barinda bir sayac artar; underwater barlar ayni
    # sayac degerini tasir -> ardisik underwater bloklari dogal olarak
    # gruplanir.
    group_id = (~underwater).cumsum()

    idx = cumulative.index
    records = []
    for _, grp in drawdown[underwater].groupby(group_id[underwater]):
        first_pos = idx.get_loc(grp.index[0])
        last_pos = idx.get_loc(grp.index[-1])
        start = idx[first_pos - 1] if first_pos > 0 else idx[0]
        valley = grp.idxmin()
        depth = float(grp.min())
        recovery = idx[last_pos + 1] if last_pos + 1 < len(idx) else None
        records.append({"start": start, "valley": valley, "recovery": recovery, "depth": depth})

    table = pd.DataFrame(records, columns=columns)
    table = table.sort_values("depth").reset_index(drop=True)
    return table.head(top_n)


# ---------------------------------------------------------------------------
# Render: Markdown
# ---------------------------------------------------------------------------


def _render_summary_rows_markdown(summary: dict) -> list[str]:
    s = summary
    rows = [
        ("Toplam Getiri", _fmt_pct(s["total_return"])),
        ("Yillik Getiri (CAGR)", _fmt_pct(s["cagr"])),
        ("Yillik Volatilite", _fmt_pct(s["ann_vol"])),
        ("Sharpe Orani", _fmt_num(s["sharpe"], 3)),
        ("Sortino Orani", _fmt_num(s["sortino"], 3)),
        ("Calmar Orani", _fmt_num(s["calmar"], 3)),
        ("Maksimum Drawdown", _fmt_pct(s["max_dd"])),
        ("Hit Rate", _fmt_pct(s["hit_rate"])),
        ("Profit Factor", _fmt_num(s["profit_factor"], 3)),
    ]
    lines = ["| Metrik | Deger |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return lines


def _render_monthly_table_markdown(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return ["Yeterli veri yok."]
    lines = ["| Yil | " + " | ".join(_MONTH_LABELS) + " | Yil Toplam |", "|---" * 14 + "|"]
    yearly = (1.0 + table.fillna(0.0)).prod(axis=1) - 1.0
    for year, row in table.iterrows():
        cells = [f"{v:.2%}" if pd.notna(v) else "-" for v in row]
        lines.append(f"| {int(year)} | " + " | ".join(cells) + f" | {yearly.loc[year]:.2%} |")
    return lines


def _render_drawdown_table_markdown(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return ["Drawdown donemi bulunamadi."]
    lines = ["| # | Baslangic | Dip | Toparlanma | Derinlik |", "|---|---|---|---|---|"]
    for i, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {_fmt_date(row.start)} | {_fmt_date(row.valley)} | "
            f"{_fmt_date(row.recovery) if row.recovery is not None else '(toparlanmadi)'} | "
            f"{row.depth:.2%} |"
        )
    return lines


def _render_markdown(data: dict) -> str:
    lines = [
        "# Performans Karnesi (Tearsheet)",
        "",
        "*Bu rapor `quanttro.reports.tearsheet.tearsheet` tarafindan otomatik uretildi.*",
        "",
        "## Ozet",
        "",
    ]
    lines += _render_summary_rows_markdown(data["summary"])

    if data["optional_rows"]:
        lines += ["", "## Ek Risk/Performans Metrikleri", "", "| Metrik | Deger |", "|---|---|"]
        lines += [f"| {label} | {value} |" for label, value in data["optional_rows"]]

    if data["benchmark_rows"]:
        lines += ["", "## Benchmark Karsilastirmasi", "", "| Metrik | Deger |", "|---|---|"]
        lines += [f"| {label} | {value} |" for label, value in data["benchmark_rows"]]

    lines += ["", "## Yillik Getiriler", ""]
    yearly_table = data["yearly_table"]
    if yearly_table.empty:
        lines.append("Yeterli veri yok.")
    else:
        lines += ["| Yil | Getiri |", "|---|---|"]
        lines += [f"| {int(year)} | {value:.2%} |" for year, value in yearly_table.items()]

    lines += ["", "## Aylik Getiri Isi Haritasi", ""]
    lines += _render_monthly_table_markdown(data["monthly_table"])

    lines += ["", "## En Buyuk Drawdown Donemleri", ""]
    lines += _render_drawdown_table_markdown(data["drawdown_table"])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render: HTML
# ---------------------------------------------------------------------------


def _heatmap_cell_style(value: float) -> str:
    if value is None or pd.isna(value):
        return "background-color:#f5f5f5;color:#999;"
    intensity = min(abs(value) / 0.10, 1.0)  # +-%10 civarinda doyuma ulasir
    alpha = 0.10 + 0.55 * intensity
    if value >= 0:
        return f"background-color:rgba(34,139,34,{alpha:.2f});"
    return f"background-color:rgba(220,20,60,{alpha:.2f});"


def _render_summary_rows_html(summary: dict) -> str:
    s = summary
    rows = [
        ("Toplam Getiri", _fmt_pct(s["total_return"])),
        ("Yillik Getiri (CAGR)", _fmt_pct(s["cagr"])),
        ("Yillik Volatilite", _fmt_pct(s["ann_vol"])),
        ("Sharpe Orani", _fmt_num(s["sharpe"], 3)),
        ("Sortino Orani", _fmt_num(s["sortino"], 3)),
        ("Calmar Orani", _fmt_num(s["calmar"], 3)),
        ("Maksimum Drawdown", _fmt_pct(s["max_dd"])),
        ("Hit Rate", _fmt_pct(s["hit_rate"])),
        ("Profit Factor", _fmt_num(s["profit_factor"], 3)),
    ]
    body = "\n".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return f"<table><tr><th>Metrik</th><th>Deger</th></tr>\n{body}\n</table>"


def _render_kv_table_html(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return f"<table><tr><th>Metrik</th><th>Deger</th></tr>\n{body}\n</table>"


def _render_monthly_table_html(table: pd.DataFrame) -> str:
    if table.empty:
        return "<p>Yeterli veri yok.</p>"
    yearly = (1.0 + table.fillna(0.0)).prod(axis=1) - 1.0
    header = "<tr><th>Yil</th>" + "".join(f"<th>{m}</th>" for m in _MONTH_LABELS) + "<th>Yil Toplam</th></tr>"
    rows = []
    for year, row in table.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("<td style='background-color:#f5f5f5;color:#999;'>-</td>")
            else:
                cells.append(f"<td style=\"{_heatmap_cell_style(v)}\">{v:.2%}</td>")
        total = yearly.loc[year]
        rows.append(
            f"<tr><td><b>{int(year)}</b></td>" + "".join(cells) +
            f"<td style=\"{_heatmap_cell_style(total)}\"><b>{total:.2%}</b></td></tr>"
        )
    return f"<table>{header}\n" + "\n".join(rows) + "\n</table>"


def _render_drawdown_table_html(table: pd.DataFrame) -> str:
    if table.empty:
        return "<p>Drawdown donemi bulunamadi.</p>"
    header = "<tr><th>#</th><th>Baslangic</th><th>Dip</th><th>Toparlanma</th><th>Derinlik</th></tr>"
    rows = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        recovery = _fmt_date(row.recovery) if row.recovery is not None else "(toparlanmadi)"
        rows.append(
            f"<tr><td>{i}</td><td>{_fmt_date(row.start)}</td><td>{_fmt_date(row.valley)}</td>"
            f"<td>{recovery}</td><td>{row.depth:.2%}</td></tr>"
        )
    return f"<table>{header}\n" + "\n".join(rows) + "\n</table>"


def _render_html(data: dict) -> str:
    optional_section = ""
    if data["optional_rows"]:
        optional_section = f"<h2>Ek Risk/Performans Metrikleri</h2>\n{_render_kv_table_html(data['optional_rows'])}"

    benchmark_section = ""
    if data["benchmark_rows"]:
        benchmark_section = f"<h2>Benchmark Karsilastirmasi</h2>\n{_render_kv_table_html(data['benchmark_rows'])}"

    yearly_table = data["yearly_table"]
    if yearly_table.empty:
        yearly_html = "<p>Yeterli veri yok.</p>"
    else:
        rows = "\n".join(f"<tr><td>{int(year)}</td><td>{value:.2%}</td></tr>" for year, value in yearly_table.items())
        yearly_html = f"<table><tr><th>Yil</th><th>Getiri</th></tr>\n{rows}\n</table>"

    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Performans Karnesi (Tearsheet)</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 960px; margin: 2rem auto; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f2f2f2; }}
  .note {{ color: #666; font-size: .9rem; margin-top: 2rem; border-top: 1px solid #ddd; padding-top: .8rem; }}
</style>
</head>
<body>
<h1>Performans Karnesi (Tearsheet)</h1>
<p><em>Bu rapor quanttro.reports.tearsheet.tearsheet tarafindan otomatik uretildi.</em></p>

<h2>Ozet</h2>
{_render_summary_rows_html(data['summary'])}

{optional_section}

{benchmark_section}

<h2>Yillik Getiriler</h2>
{yearly_html}

<h2>Aylik Getiri Isi Haritasi</h2>
{_render_monthly_table_html(data['monthly_table'])}

<h2>En Buyuk Drawdown Donemleri</h2>
{_render_drawdown_table_html(data['drawdown_table'])}

<p class="note">Not: Gecmis performans gelecegi garanti etmez. Bu rapor
`quanttro.backtest.metrics` icindeki fonksiyonlari kullanir; bazi ek
metrikler (omega ratio, VaR/CVaR, alpha/beta vb.) o modulde henuz
eklenmemisse tablolardan dusurulur.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public: tearsheet
# ---------------------------------------------------------------------------


def tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    output_path: str | None = None,
    format: str = "html",
) -> str:
    """QuantStats tarzi, kapsamli bir performans karnesi (tearsheet) uretir.

    Ozet tablo (toplam getiri, CAGR, yillik volatilite, Sharpe, Sortino,
    Calmar, maksimum drawdown, hit rate, profit factor - hepsi
    `quanttro.backtest.metrics` fonksiyonlari ile hesaplanir), varsa
    (paralel gelistirilen) ek risk/performans metrikleri, varsa benchmark
    karsilastirmasi (alpha/beta, capture ratio), yillik getiriler, aylik
    getiri isi haritasi ve en buyuk drawdown donemlerini icerir.

    `quanttro.backtest.metrics` modulundeki ek metrikler (omega_ratio,
    tail_ratio, value_at_risk, conditional_value_at_risk, alpha_beta,
    capture_ratio, kelly_criterion, risk_of_ruin, ulcer_index,
    common_sense_ratio, avg_win, avg_loss, payoff_ratio, recovery_factor,
    gain_to_pain_ratio) HENUZ mevcut olmasa (ya da farkli bir imzaya sahip
    olsa) bile bu fonksiyon HATASIZ calisir - ilgili satirlar rapordan
    dusurulur.

    Parametreler
    ----------
    returns : pd.Series
        DatetimeIndex'li basit (aritmetik) getiri serisi.
    benchmark_returns : pd.Series veya None
        Verilirse, ortak tarih araligi uzerinde bir "Benchmark
        Karsilastirmasi" bolumu eklenir (korelasyon, benchmark toplam
        getirisi ve varsa alpha/beta, capture ratio).
    output_path : str veya None
        Verilirse rapor bu yola yazilir (ust dizinler otomatik olusturulur).
        Verilmezse sadece string olarak dondurulur.
    format : str
        "html" (varsayilan) ya da "markdown".

    Donus
    -----
    str
        Uretilen rapor icerigi (``output_path`` verilse de verilmese de
        dondurulur).

    Ornek
    -----
    >>> rpt = tearsheet(returns, output_path="reports_out/tearsheet.html")
    """
    if format not in ("html", "markdown"):
        raise ValueError(f"Desteklenmeyen format: {format!r} ('html' ya da 'markdown' olmali)")

    r = _clean_returns(returns)
    summary = _core_summary(r)
    optional_rows = _collect_optional_metrics(r, summary["cumulative"])

    benchmark_rows: list[tuple[str, str]] = []
    if benchmark_returns is not None:
        b = _clean_returns(benchmark_returns)
        if not b.empty:
            benchmark_rows = _collect_benchmark_section(r, b)

    monthly_table = monthly_returns_table(r)
    if monthly_table.empty:
        yearly_table = pd.Series(dtype=float)
    else:
        yearly_table = (1.0 + monthly_table.fillna(0.0)).prod(axis=1) - 1.0

    data = {
        "summary": summary,
        "optional_rows": optional_rows,
        "benchmark_rows": benchmark_rows,
        "monthly_table": monthly_table,
        "yearly_table": yearly_table,
        "drawdown_table": drawdown_periods(r, top_n=5),
    }

    content = _render_markdown(data) if format == "markdown" else _render_html(data)
    disclaimer = (
        "\n\n---\n*Bu rapor quanttro tarafindan otomatik uretilmistir. "
        "Yatirim tavsiyesi degildir; gecmis performans gelecegi garanti etmez.*\n"
        if format == "markdown"
        else '\n<hr><p style="font-size:12px;color:#888;">'
        "This report was generated automatically by quanttro. "
        "Not investment advice; past performance does not guarantee future results.</p>\n"
    )
    content += disclaimer

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    return content
