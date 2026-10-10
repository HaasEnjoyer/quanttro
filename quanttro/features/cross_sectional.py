"""
quanttro.features.cross_sectional
======================================
Coklu-hisse (panel) bazli bagil guc / siralama ozellikleri.

Bu modulun fonksiyonlari "wide" formatta girdi alir: index = zaman damgasi,
kolonlar = sembol (ticker), degerler = o sembolun o zamandaki getirisi/
degeri. Bu, `feature_engineering.py` icindeki `add_relative_strength` ve
`add_relative_target` fonksiyonlarinin (dict[sembol -> DataFrame] uzerinde
calisan) ayni mantigini daha genel/panel-dostu bir bicimde ifade eder.

Neden "bagil" (relative)? Piyasa-geneli (veya sektor-geneli) ortak trend -
orn. TL'nin enflasyon/devaluasyon trendi - BIST30 icindeki TUM hisselerde
paylasilan ortak bir bilesendir. Ham getiriyi/hedefi oldugu gibi kullanmak
modele sahte bir "hep yukari" sinyali ogretir; piyasaya (veya sektore) GORE
bagil deger kullanmak bu ortak bilesenin yarattigi sahte korelasyonu giderir.
"""

from __future__ import annotations

import pandas as pd


def _demean_cross_section(values_by_symbol: pd.DataFrame) -> pd.DataFrame:
    """Her zaman noktasinda (satir) tum sembollerin ortalamasini cikarir."""
    market_mean = values_by_symbol.mean(axis=1)
    return values_by_symbol.sub(market_mean, axis=0)


def relative_strength(returns_by_symbol: pd.DataFrame) -> pd.DataFrame:
    """
    Her hissenin, o anki piyasa (tum semboller) ortalama getirisine gore
    bagil getirisi: ret_sembol - ortalama(ret_tum_semboller), satir bazinda.

    `feature_engineering.py` icindeki `add_relative_strength` fonksiyonunun
    (dict uzerinde calisan) genellestirilmis/wide-format hali.
    """
    return _demean_cross_section(returns_by_symbol)


def relative_forward_target(forward_returns_by_symbol: pd.DataFrame) -> pd.DataFrame:
    """
    Ileri (forward/target) getirilerin piyasa ortalamasina gore bagil hali.

    `feature_engineering.py` icindeki `add_relative_target` fonksiyonunun
    genellestirilmis hali: piyasa-geneli enflasyon/devaluasyon trendi hedef
    degiskenden cikarilir, boylece model "hep yukari" sahte sinyalini
    ogrenmez.
    """
    return _demean_cross_section(forward_returns_by_symbol)


def cross_sectional_rank(values_by_symbol: pd.DataFrame) -> pd.DataFrame:
    """
    Her zaman noktasinda (satir), sembolleri degere gore [0, 1] araliginda
    percentile olarak siralar (en kucuk deger ~1/n'e, en buyuk deger 1.0'a
    yakin olur; esit degerler ortalama rank alir).

    NaN degerler o satirin siralamasina dahil edilmez - percentile, o
    satirdaki gecerli (non-NaN) gozlem sayisina gore hesaplanir.
    """
    return values_by_symbol.rank(axis=1, pct=True)


def sector_relative_strength(
    returns_by_symbol: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Her hissenin KENDI SEKTORUNUN ortalama getirisine gore bagil getirisi.

    `sector_map` (sembol -> sektor adi) verilmezse veya bossa, dogrudan
    `relative_strength` (tum piyasaya gore bagil guc) sonucuna geri duser.

    `sector_map` verildiginde:
      - Sektoru bilinen VE kendi sektorunde en az 2 sembol bulunan hisseler
        kendi sektor ortalamasina gore hesaplanir.
      - Sektoru `sector_map`'te olmayan VEYA kendi sektorunde yalniz basina
        olan (tek sembollu sektor icin "bagil guc" tanimsiz/anlamsizdir)
        hisseler, piyasa ortalamasina gore bagil guce (fallback) duser.

    Donus: `returns_by_symbol` ile ayni index/kolonlara (ayni sirada) sahip
    pd.DataFrame.
    """
    if not sector_map:
        return relative_strength(returns_by_symbol)

    market_mean = returns_by_symbol.mean(axis=1)
    result = pd.DataFrame(index=returns_by_symbol.index, columns=returns_by_symbol.columns, dtype=float)

    groups: dict[str | None, list[str]] = {}
    for symbol in returns_by_symbol.columns:
        sector = sector_map.get(symbol)
        groups.setdefault(sector, []).append(symbol)

    for sector, symbols in groups.items():
        if sector is None or len(symbols) < 2:
            # sektoru bilinmeyen VEYA tek-sembollu sektor: piyasa ortalamasina fallback
            for symbol in symbols:
                result[symbol] = returns_by_symbol[symbol] - market_mean
            continue
        sector_mean = returns_by_symbol[symbols].mean(axis=1)
        for symbol in symbols:
            result[symbol] = returns_by_symbol[symbol] - sector_mean

    return result[returns_by_symbol.columns]
