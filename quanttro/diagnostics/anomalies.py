"""Tavan/taban (limit-up/limit-down) anomalilerini otomatik tespit eden fonksiyonlar.

Bu modul, KONTR hissesinde elle kesfedilen "8 gunluk tavan serisi" ve "digerlerinden
3.7x yuksek tavan+taban orani" gibi bulgulari genel, tekrar kullanilabilir fonksiyonlara
donusturur. Varsayilan esik (0.095), BIST'teki %10 tavan/taban bandina karsilik gelir;
feature_engineering.py'deki LIMIT_THRESHOLD ile ayni mantiktir (yuvarlama/kayma payi
icin %10 degil %9.5 kullanilir).

Onemli: bu fonksiyonlar "ongorulebilirlik" iddia etmez. KONTR arastirmasinin gosterdigi
gibi, yuksek otokorelasyon/limit-seri sikligi dusuk-float donemlerde manipulasyon ya da
distress sinyali olabilir (bkz. docs/KNOWN_PITFALLS.md). Bu modul sadece "bu patern var
mi, ne kadar siddetli" sorusuna otomatik cevap verir; yorumu kullaniciya birakir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT_THRESHOLD = 0.095  # BIST %10 tavan/taban icin guvenlik payli esik (feature_engineering.py ile uyumlu)


def detect_limit_streaks(
    returns: pd.Series,
    threshold: float = LIMIT_THRESHOLD,
    min_streak: int = 2,
) -> pd.DataFrame:
    """Ardisik tavan (+limit) ya da taban (-limit) gunlerinden olusan serileri bulur.

    KONTR analizinde elle bulunan "8 gunluk tavan serisi" gibi olaylari otomatik
    tespit etmek icin yazildi: `returns` serisinde ayni yonde (hep tavan ya da hep
    taban) ust uste gelen, en az `min_streak` uzunlugundaki tum bloklari dondurur.

    Parametreler
    ----------
    returns : pd.Series
        Gunluk (ya da baska bir periyodik) getiri serisi, DatetimeIndex ile.
    threshold : float
        |getiri| >= threshold olan gunler "limitte" sayilir.
    min_streak : int
        Raporlanacak minimum ardisik gun sayisi (varsayilan 2 - yani en az 2 gun ust uste).

    Donus
    -----
    pd.DataFrame
        Kolonlar: start_date, end_date, length, direction ("tavan" | "taban").
        Hic seri bulunmazsa bos (ama kolonlari dogru) bir DataFrame doner.
    """
    s = returns.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "length", "direction"])

    sign = np.sign(s)
    at_limit = s.abs() >= threshold
    # Her gun icin durum: +1 (tavan), -1 (taban), 0 (limit disi)
    state = sign.where(at_limit, 0)

    # Durum degistigi her noktada yeni bir "blok" baslar; ayni durumun ardisik
    # tekrarlari ayni bloga duser (groupby ile ardisik run'lari ayiklamanin standart yolu)
    block_id = (state != state.shift()).cumsum()

    rows = []
    for _, block in s.groupby(block_id):
        block_state = state.loc[block.index[0]]
        if block_state == 0:
            continue
        length = len(block)
        if length < min_streak:
            continue
        rows.append({
            "start_date": block.index[0],
            "end_date": block.index[-1],
            "length": length,
            "direction": "tavan" if block_state > 0 else "taban",
        })

    out = pd.DataFrame(rows, columns=["start_date", "end_date", "length", "direction"])
    if not out.empty:
        out = out.sort_values("length", ascending=False).reset_index(drop=True)
    return out


def limit_frequency_by_year(
    returns: pd.Series,
    threshold: float = LIMIT_THRESHOLD,
) -> pd.DataFrame:
    """Yil bazinda tavan/taban gunlerinin oranini hesaplar.

    KONTR analizinde "2020'de %24.5, 2026'da %11" seklinde bulunan yillik tavan/taban
    oranlarini otomatik uretmek icin yazildi. Bir hissenin rejiminin zaman icinde nasil
    degistigini gormek icin (ornegin IPO sonrasi dusuk-float donemi ile daha sonraki
    yuksek-float donemi karsilastirmak) kullanilir.

    Donus
    -----
    pd.DataFrame
        Kolonlar: year, n_gun, n_tavan, n_taban, tavan_orani, taban_orani,
        toplam_limit_orani. Yillara gore artan sirada.
    """
    s = returns.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=[
            "year", "n_gun", "n_tavan", "n_taban",
            "tavan_orani", "taban_orani", "toplam_limit_orani",
        ])

    frame = pd.DataFrame({
        "tavan": s >= threshold,
        "taban": s <= -threshold,
    })
    frame["year"] = frame.index.year

    grouped = frame.groupby("year").agg(
        n_gun=("tavan", "size"),
        n_tavan=("tavan", "sum"),
        n_taban=("taban", "sum"),
    )
    grouped["tavan_orani"] = grouped["n_tavan"] / grouped["n_gun"]
    grouped["taban_orani"] = grouped["n_taban"] / grouped["n_gun"]
    grouped["toplam_limit_orani"] = grouped["tavan_orani"] + grouped["taban_orani"]
    return grouped.reset_index()


def flag_outlier_stocks(
    symbol_stats: dict[str, pd.DataFrame],
    combined_rate_threshold: float = 0.05,
    return_col: str = "ret_1",
    threshold: float = LIMIT_THRESHOLD,
    lookback_bars: int = 252,
    relative_multiplier: float = 3.0,
) -> list[str]:
    """Son donemde tavan+taban orani anormal yuksek olan hisseleri isaretler.

    KONTR'u digerlerinden ~3.7x ayiran mantigi genellestirir: her hissenin son
    `lookback_bars` (varsayilan ~1 yil) gunundeki tavan+taban oranini hesaplar, ve
    ya (a) bu oran mutlak esigi (`combined_rate_threshold`) gecen, ya da (b) tum
    hisselerin medyaninin `relative_multiplier` katindan fazla olan hisseleri
    "outlier" olarak isaretler.

    Parametreler
    ----------
    symbol_stats : dict[str, pd.DataFrame]
        {"KONTR": df, "THYAO": df, ...} - her df en az `return_col` kolonunu
        (gunluk getiri) icermeli, DatetimeIndex ile sirali olmali.
    combined_rate_threshold : float
        Mutlak esik: tavan+taban orani bunu gecen hisse direkt isaretlenir.
    return_col : str
        Getiri kolonunun adi (ozellik tablolarinda "ret_1").
    threshold : float
        Tavan/taban sayilma esigi.
    lookback_bars : int
        Degerlendirilecek son bar sayisi (varsayilan 252 ~ 1 is yili).
    relative_multiplier : float
        Medyanin kac katini gecen hisselerin de "outlier" sayilacagini belirler.

    Donus
    -----
    list[str]
        Isaretlenen sembollerin alfabetik sirali listesi. Hesaplanamayan (veri yok/
        kolon yok) semboller sessizce atlanir.
    """
    rates: dict[str, float] = {}
    for symbol, df in symbol_stats.items():
        if df is None or df.empty or return_col not in df.columns:
            continue
        recent = df[return_col].dropna().iloc[-lookback_bars:]
        if recent.empty:
            continue
        rates[symbol] = float((recent.abs() >= threshold).mean())

    if not rates:
        return []

    rate_series = pd.Series(rates)
    median_rate = rate_series.median()

    flagged = []
    for symbol, rate in rate_series.items():
        exceeds_absolute = rate >= combined_rate_threshold
        exceeds_relative = median_rate > 0 and rate >= relative_multiplier * median_rate
        if exceeds_absolute or exceeds_relative:
            flagged.append(symbol)

    return sorted(flagged)
