"""
quanttro.portfolio.allocation
==================================
Hierarchical Risk Parity (HRP) agirliklandirma - Lopez de Prado (2016),
"Building Diversified Portfolios that Outperform Out-of-Sample" - ve
surekli agirliklari gercek tam-sayi hisse/lot adedine ceviren discrete
allocation.

HRP, klasik ortalama-varyans optimizasyonunun (``quanttro.portfolio.
optimize.mean_variance_optimize``) aksine kovaryans matrisinin TERSINI
ALMAZ; bunun yerine hiyerarsik kumeleme agacina (dendrogram) gore
yukaridan-asagiya (top-down) risk dagitir. Bu, kovaryans matrisi kotu
kosullu (ill-conditioned) oldugunda - orn. varlik sayisi gozlem sayisina
yakin/fazla oldugunda ya da varliklar arasi korelasyon cok yuksek oldugunda
- mean-variance'tan daha SAGLAM (robust) agirliklar uretir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _quasi_diagonalize(link: np.ndarray) -> list[int]:
    """
    Lopez de Prado'nun "get_quasi_diag" fonksiyonu: `linkage` agacini,
    kumeleme sirasinda birlesen (dolayisiyla en benzer) varliklari
    birbirine YAKIN siralayan tek bir permutasyona (orijinal varlik
    indeksleri uzerinde) donusturur.

    Sonuc permutasyonla yeniden siralanan kovaryans/korelasyon matrisi
    "quasi-diagonal" olur: en benzer varliklar kosegene yakin bloklar
    halinde kumelenir. Bu siralama, recursive bisection'in (bkz.
    `_recursive_bisection`) benzer varliklari ayni alt-kumede tutarak
    anlamli bir sekilde ikiye bolebilmesi icin gereklidir.
    """
    link = link.astype(int)
    num_items = link[-1, 3]  # linkage'in son satirindaki eleman sayisi = orijinal varlik sayisi
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # yeni cocuklar icin index'lerde bosluk ac
        clusters = sort_ix[sort_ix >= num_items]  # halen bir "kume" (cluster) olan elemanlar
        i = clusters.index
        j = clusters.to_numpy() - num_items
        sort_ix[i] = link[j, 0]  # kumenin ilk (sol) cocugu, mevcut yerine yazilir
        second_children = pd.Series(link[j, 1], index=i + 1)  # kumenin ikinci (sag) cocugu
        sort_ix = pd.concat([sort_ix, second_children])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """
    Bir varlik alt-kumesinin (cluster) TERS-VARYANS agirlikli (inverse-
    variance weighted) portfoy varyansi.

    Ters-varyans agirliklari, tam kovaryans matrisinin tersini almadan
    (sadece kosegen/diagonal kullanilarak) hesaplanan basit ama HRP'nin
    ic-kume (intra-cluster) agirliklandirmasinda kullandigi standart
    yaklasimdir.
    """
    cov_slice = cov.loc[items, items].to_numpy(dtype=float)
    inv_var = 1.0 / np.diag(cov_slice)
    inv_var /= inv_var.sum()
    w = inv_var.reshape(-1, 1)
    return float((w.T @ cov_slice @ w)[0, 0])


def _recursive_bisection(cov: pd.DataFrame, sorted_symbols: list[str]) -> pd.Series:
    """
    Lopez de Prado'nun "get_rec_bisection" fonksiyonu: quasi-diagonal
    sirali varlik listesini yukaridan-asagiya (top-down) ikiye bole bole,
    her adimda iki alt-kumeye, kume varyanslarinin TERSI ile orantili
    (yani daha riskli alt-kumeye daha az) agirlik verir::

        alpha = 1 - var(sol_kume) / (var(sol_kume) + var(sag_kume))
        agirlik(sol_kume) *= alpha
        agirlik(sag_kume) *= (1 - alpha)

    Her alt-kume kendi icinde tek bir varlik kalana kadar ikiye bolunmeye
    devam eder; nihai agirlik, bir varligin tum ust-kumelerden miras aldigi
    carpanlarin birikimli carpimidir.
    """
    weights = pd.Series(1.0, index=sorted_symbols)
    clusters = [sorted_symbols]

    while len(clusters) > 0:
        # her kumeyi (birden fazla elemani varsa) yaklasik ortadan ikiye bol
        clusters = [
            cluster[start:end]
            for cluster in clusters
            for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            left = clusters[i]
            right = clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    return weights


def hrp_optimize(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (HRP) agirliklandirma - Lopez de Prado (2016).

    Algoritma (orijinal HRP'nin 4 adimi):

    1) Getiri korelasyon matrisinden bir MESAFE matrisi turetilir:
       ``d_ij = sqrt((1 - corr_ij) / 2)`` (korelasyon +1 -> mesafe 0,
       korelasyon -1 -> mesafe 1).
    2) Bu mesafe matrisi `scipy.cluster.hierarchy.linkage` (single-linkage)
       ile hiyerarsik bir kumeleme agacina (dendrogram) donusturulur.
    3) Quasi-diagonalizasyon (`_quasi_diagonalize`): agacin yapraklari,
       benzer varliklari birbirine yakin siralayan tek bir permutasyona
       donusturulur.
    4) Recursive bisection (`_recursive_bisection`): bu sirali liste
       yukaridan-asagiya ikiye bolune bolune, her alt-kumeye kume
       varyansinin tersiyle orantili agirlik verilir.

    Mean-variance optimizasyonunun aksine kovaryans matrisinin tersi
    ALINMAZ - bu yuzden varlik sayisi/korelasyon yapisi mean-variance'i
    dengesiz hale getirecek durumlarda bile SAGLAM agirliklar uretir.

    Donus: `returns` kolonlariyla (sembollerle) AYNI SIRADA, toplami 1
    olan, NEGATIF OLMAYAN agirlik Series'i.
    """
    clean = returns.dropna(how="any")
    symbols = list(clean.columns)
    n = len(symbols)
    if n < 2:
        raise ValueError("hrp_optimize en az 2 varlik (kolon) gerektirir")
    if len(clean) < 2:
        raise ValueError("hrp_optimize icin en az 2 gecerli (NaN'siz) gozlem (satir) gerekir")

    corr = clean.corr()
    cov = clean.cov()

    distance = np.sqrt(np.clip((1.0 - corr.to_numpy(dtype=float)) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    # kayan nokta yuvarlamasiyla bozulabilecek simetriyi garantiye al (squareform bunu zorunlu kilar)
    distance = (distance + distance.T) / 2.0
    condensed_distance = squareform(distance, checks=False)
    link = linkage(condensed_distance, method="single")

    sort_ix = _quasi_diagonalize(link)
    sorted_symbols = [symbols[i] for i in sort_ix]

    weights = _recursive_bisection(cov, sorted_symbols)
    weights = weights.reindex(symbols)  # orijinal kolon sirasina geri don
    weights = weights / weights.sum()
    weights.name = "weight"
    return weights


def discrete_allocation(weights: pd.Series, latest_prices: pd.Series, total_value: float) -> dict:
    """
    Surekli (continuous) portfoy agirliklarini gercek tam-sayi hisse/lot
    adedine cevirir (greedy algoritma).

    Algoritma:
    1) Semboller agirliga gore BUYUKTEN KUCUGE siralanir.
    2) Her sembol icin, o sembolun "ideal" dolar tahsisi
       (``agirlik * total_value``) hesaplanir ve bu tutarla alinabilecek
       MAKSIMUM TAM lot sayisi (``floor(ideal_dolar / fiyat)``) bulunur;
       bu sayi, o ana kadar harcanmamis KALAN NAKTI asamayacak sekilde
       sinirlanir (guvenlik tavani - yuvarlama hatalari harcamanin
       `total_value`'yu asmasina izin vermez).
    3) Alinan lotun bedeli kalan nakitten dusulur, siradaki (bir sonraki
       en yuksek agirlikli) sembole gecilir.

    `weights` toplami 1 olmayabilir (orn. kismi/eksik bir portfoy)
    - fonksiyon basta agirliklari kendi toplamina bolerek normalize eder.

    Donus: ``{"allocation": {sembol: lot_sayisi, ...}, "leftover_cash": float}``
    - ``allocation`` sadece en az 1 lot alinabilen sembolleri icerir;
      ``leftover_cash`` tahsis edilemeyen (hicbir sembolde tam lota
      yetmeyen) kalan nakittir.
    """
    if total_value <= 0:
        raise ValueError("total_value pozitif olmalidir")

    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("weights toplami pozitif olmalidir")
    normalized_weights = weights / weight_sum

    prices = latest_prices.reindex(normalized_weights.index)
    missing = prices[prices.isna()].index.tolist()
    if missing:
        raise ValueError(f"latest_prices icinde fiyati olmayan semboller: {missing}")
    if (prices <= 0).any():
        bad = prices[prices <= 0].index.tolist()
        raise ValueError(f"latest_prices pozitif olmalidir, sorunlu semboller: {bad}")

    order = normalized_weights.sort_values(ascending=False).index

    allocation: dict[str, int] = {}
    remaining_cash = float(total_value)
    for symbol in order:
        price = float(prices[symbol])
        ideal_dollars = float(normalized_weights[symbol]) * total_value
        n_shares = int(ideal_dollars // price)
        n_shares = min(n_shares, int(remaining_cash // price))
        n_shares = max(n_shares, 0)
        if n_shares > 0:
            allocation[symbol] = n_shares
            remaining_cash -= n_shares * price

    return {"allocation": allocation, "leftover_cash": float(remaining_cash)}
