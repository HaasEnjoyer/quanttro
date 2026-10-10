"""Risk ve performans metrikleri.

Bu oturumda dashboard uzerinde konusulan ama otomatiklestirilmemis
metrikler: Sharpe, Sortino, Max Drawdown, Calmar, hit rate, profit factor
ve Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Tum fonksiyonlar periyot bazli (gunluk/saatlik/vb., ``periods_per_year``
parametresiyle belirtilen periyot sikligina gore) basit (aritmetik,
compound DEGIL) getiri serileri uzerinde calisir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _as_series(values) -> pd.Series:
    return pd.Series(values, dtype=float).dropna()


def sharpe_ratio(returns, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Yillliklandirilmis Sharpe orani.

    ``Sharpe = mean(r_excess) / std(r_excess, ddof=1) * sqrt(periods_per_year)``

    ``risk_free`` YILLIK risksiz oran olarak verilir; periyot bazina
    (``risk_free / periods_per_year``) bolunerek getirilerden cikarilir.
    Standart sapma ornek (ddof=1) standart sapmasidir. Getiri sayisi 2'den
    az ya da standart sapma 0 ise NaN dondurur.
    """
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - (risk_free / periods_per_year)
    std = excess.std(ddof=1)
    # Not: tam ayni (bit-bit esit) degerlerden olusan bir seride bile
    # std() kayan nokta yuvarlama hatasiyla tam 0 degil, ~1e-18 gibi
    # ihmal edilebilir bir deger donebilir; bu yuzden "sifira yakin"
    # kontrolu tolerans ile yapilir (atol=1e-12, gercek getiri
    # oynakliklarindan kat kat kucuk).
    if not np.isfinite(std) or np.isclose(std, 0.0, atol=1e-12):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Yillliklandirilmis Sortino orani: Sharpe'tan farki, paydada sadece
    asagi yonlu (negatif) sapmalarin kullanildigi "downside deviation"
    olmasidir (yukari yonlu oynaklik cezalandirilmaz).

    ``downside_dev = sqrt( mean( min(r_excess, 0)^2 ) )``  (TUM periyotlar
    uzerinden - sadece negatif periyotlar degil, paydaya sifir olarak
    girerler, bu standart Sortino tanimidir)

    ``Sortino = mean(r_excess) / downside_dev * sqrt(periods_per_year)``
    """
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - (risk_free / periods_per_year)
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if not np.isfinite(downside_dev) or np.isclose(downside_dev, 0.0, atol=1e-12):
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns) -> float:
    """Maksimum drawdown: kumulatif getiri/varlik egrisinin (orn.
    ``(1 + r).cumprod()``) o ana kadarki tepe degerine (running max) gore
    en kotu (en negatif) yuzdesel geri cekilmesi.

    ``drawdown_t = curve_t / running_max_t - 1``, dondurulen deger
    ``min(drawdown_t)``'dir ve NEGATIFTIR (orn. -0.23 -> %23 maksimum
    dusus). Bos seri icin NaN dondurur.
    """
    curve = _as_series(cumulative_returns)
    if len(curve) == 0:
        return float("nan")
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    """Calmar orani = Yillliklandirilmis getiri (CAGR) / |Maksimum drawdown|.

    Getiri serisinden kumulatif buyume egrisi (``(1 + r).cumprod()``)
    kurulur; CAGR bu egrinin toplam suresine gore yillliklandirilmis
    buyume oranidir, maksimum drawdown da ayni egri uzerinden
    hesaplanir (bkz. :func:`max_drawdown`).
    """
    r = _as_series(returns)
    n = len(r)
    if n < 2:
        return float("nan")
    cumulative = (1.0 + r).cumprod()
    mdd = max_drawdown(cumulative)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    total_growth = cumulative.iloc[-1]
    if total_growth <= 0:
        return float("nan")
    annualized_return = total_growth ** (periods_per_year / n) - 1.0
    return float(annualized_return / abs(mdd))


def hit_rate(returns) -> float:
    """Pozitif getiri orani: getirinin 0'dan buyuk oldugu periyotlarin
    toplam periyoda orani (0 getiri pozitif sayilmaz)."""
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def profit_factor(returns) -> float:
    """Profit factor = toplam pozitif getiri / |toplam negatif getiri|.

    Kayip yoksa ve kazanc varsa +inf, hem kazanc hem kayip yoksa (orn.
    serinin tamami sifir) NaN dondurur.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    gains = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / abs(losses))


_EULER_MASCHERONI = 0.5772156649015328606


def _sharpe_estimation_std(sharpe: float, skewness: float, kurtosis: float, n_observations: int) -> float:
    """Gozlemlenen Sharpe oraninin tahmin standart hatasi (Mertens, 2002;
    Bailey & Lopez de Prado'nun DSR calismasinda kullanilan haliyle).
    Getiri dagiliminin carpikligi (skewness) ve TAM (Pearson, normal
    dagilimda = 3) kurtosis'i de hesaba katar::

        se(SR) = sqrt( (1 - skew*SR + (kurtosis - 1) / 4 * SR^2) / (n_observations - 1) )
    """
    if n_observations < 2:
        raise ValueError("n_observations en az 2 olmalidir")
    variance = (1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2) / (n_observations - 1)
    variance = max(variance, 0.0)
    return float(np.sqrt(variance))


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    skewness: float,
    kurtosis: float,
    n_observations: int,
) -> float:
    """Deflated Sharpe Ratio (DSR) - Bailey & Lopez de Prado (2014).

    Coklu strateji/parametre denemesinin (``n_trials``) sans eseri yuksek
    Sharpe uretme riskini ve getiri dagiliminin normallikten uzakligini
    (``skewness``, ``kurtosis``) duzelterek, "gercek Sharpe orani > 0"
    olasiligini [0, 1] araliginda verir.

    Parametre tanimlari (ONEMLI):
    - ``sharpe``: degerlendirilen (n_trials deneme icinden "en iyi" secilen)
      Sharpe orani. Periyot bazli (yillliklandirilmamis) kullanilmasi
      onerilir; yillik Sharpe kullanilacaksa sonucun yorumu da ona gore
      degisir (fonksiyon iceride bir donusum YAPMAZ, girdi tutarliligi
      cagirana aittir).
    - ``kurtosis``: getiri serisinin TAM (Pearson) kurtosis degeri - normal
      dagilimda 3'tur. ``scipy.stats.kurtosis(x, fisher=False)`` ya da
      ``pandas.Series.kurt() + 3`` ile hesaplanir; pandas/scipy'nin
      varsayilan "fazla/excess" kurtosis'i (normalde 0) DEGILDIR -
      karistirilirsa DSR hatali cikar.
    - ``n_trials``: denenen strateji/parametre kombinasyonu sayisi. 1 ise
      coklu deneme duzeltmesi yapilmaz (karsilastirma esigi SR_0 = 0 kabul
      edilir, yani DSR klasik Probabilistic Sharpe Ratio'ya esit olur).

    Yontem (Bailey & Lopez de Prado, 2014):
    1) ``n_trials`` denemenin, gercekte beceri (skill) olmasa bile sans
       eseri uretebilecegi BEKLENEN MAKSIMUM Sharpe orani (SR_0),
       ekstrem deger teorisinden gelen bir yaklasimla, Sharpe tahmininin
       standart hatasi ve Euler-Mascheroni sabiti kullanilarak hesaplanir::

           SR_0 = se(SR) * [ (1-gamma)*Z^-1(1 - 1/n_trials)
                              + gamma*Z^-1(1 - 1/(n_trials*e)) ]

    2) Gozlemlenen Sharpe'in bu SR_0 esigini asma olasiligi, standart
       normal CDF araciligiyla (Probabilistic Sharpe Ratio mantigiyla)
       hesaplanir - bu deger DSR'dir::

           DSR = CDF( (SR - SR_0) / se(SR) )

    NOT: ``n_trials`` denemenin HER BIRININ kendi Sharpe tahmin varyansi
    ayri ayri bilinmedigi icin (tek bir "en iyi" SR ve sadece deneme
    sayisi verildiginden), yaygin bir yaklasimla denemeler arasindaki SR
    varyansinin vekili olarak gozlemlenen SR'nin kendi tahmin standart
    hatasi (``_sharpe_estimation_std``) kullanilir.
    """
    if n_trials < 1:
        raise ValueError("n_trials en az 1 olmalidir")

    sr_std = _sharpe_estimation_std(sharpe, skewness, kurtosis, n_observations)
    if sr_std == 0:
        return float("nan")

    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_benchmark = sr_std * ((1 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)

    z_score = (sharpe - sr_benchmark) / sr_std
    return float(norm.cdf(z_score))


# --------------------------------------------------------------------------
# Genisletilmis metrikler (QuantStats / empyrical benzeri, BIST'e ozel
# olmayan, genel kullanim icin risk/performans metrikleri)
# --------------------------------------------------------------------------


def _align_pair(returns, benchmark_returns) -> pd.DataFrame:
    """Iki getiri serisini ORTAK index uzerinde (ic kesisim) hizalar ve
    herhangi birinde NaN olan satirlari atar. Girdi duz liste/array ise
    pandas varsayilan RangeIndex ile pozisyonel hizalama yapilir."""
    r = pd.Series(returns, dtype=float)
    b = pd.Series(benchmark_returns, dtype=float)
    aligned = pd.concat([r.rename("r"), b.rename("b")], axis=1, join="inner").dropna()
    return aligned


def alpha_beta(
    returns,
    benchmark_returns,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """CAPM alfa (yillliklandirilmis) ve beta.

    Getiri serileri ortak (index kesisimi) donemler uzerinde hizalanir.

    ``beta = Cov(r_excess, b_excess) / Var(b_excess)``  (ornek kovaryans
    ve varyans, ddof=1)

    ``alpha`` once periyot bazinda hesaplanir::

        alpha_period = mean(r_excess) - beta * mean(b_excess)

    burada ``r_excess = r - risk_free/periods_per_year`` (benchmark icin
    de ayni sekilde). Yillliklandirma, bu moduldeki diger fonksiyonlarla
    (orn. Sharpe) tutarli olarak aritmetik (compound OLMAYAN) basit
    carpimla yapilir::

        alpha = alpha_period * periods_per_year

    En az 2 ortak (NaN olmayan) gozlem yoksa ya da benchmark'in (excess)
    varyansi ~0 ise ``(nan, nan)`` dondurulur.
    """
    aligned = _align_pair(returns, benchmark_returns)
    if len(aligned) < 2:
        return float("nan"), float("nan")
    rf_period = risk_free / periods_per_year
    r_excess = aligned["r"] - rf_period
    b_excess = aligned["b"] - rf_period
    b_var = b_excess.var(ddof=1)
    if not np.isfinite(b_var) or np.isclose(b_var, 0.0, atol=1e-12):
        return float("nan"), float("nan")
    covariance = float(np.cov(r_excess.to_numpy(), b_excess.to_numpy(), ddof=1)[0, 1])
    beta = covariance / b_var
    alpha_period = r_excess.mean() - beta * b_excess.mean()
    alpha = alpha_period * periods_per_year
    return float(alpha), float(beta)


def omega_ratio(returns, required_return: float = 0.0) -> float:
    """Omega orani: bir esik (``required_return``) ustundeki kazanclarin
    toplami / esik altindaki kayiplarin toplaminin mutlak degeri.

    ``excess = r - required_return``
    ``Omega = sum(excess[excess > 0]) / |sum(excess[excess < 0])|``

    ``required_return`` PERIYOT bazli bir esik getiri olarak alinir
    (yillik bir esik kullanilacaksa cagiran bunu periyot bazina
    cevirmelidir). Esik altinda hic kayip yoksa ve pay > 0 ise +inf, hem
    pay hem payda 0 ise NaN dondurur.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    excess = r - required_return
    gains = excess[excess > 0].sum()
    losses = excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / abs(losses))


def tail_ratio(returns) -> float:
    """Tail ratio = 95. persentil / |5. persentil|.

    Getiri dagiliminin sag (kazanc) kuyrugunun sol (kayip) kuyruguna
    oranidir; 1'den buyuk degerler sag kuyrugun (ekstrem kazanclarin)
    sol kuyruktan (ekstrem kayiplardan) daha "kalin"/buyuk oldugunu
    gosterir. Persentiller numpy'nin varsayilan dogrusal
    interpolasyonuyla (``np.percentile``) hesaplanir. 5. persentil 0
    ise NaN dondurur.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    p95 = np.percentile(r.to_numpy(), 95)
    p5 = np.percentile(r.to_numpy(), 5)
    if p5 == 0:
        return float("nan")
    return float(p95 / abs(p5))


def value_at_risk(returns, cutoff: float = 0.05) -> float:
    """Tarihsel Value at Risk (VaR), persentil yontemi.

    ``VaR = percentile(r, cutoff * 100)``

    Donus degeri getiri biriminde ve genellikle NEGATIFTIR (orn. -0.03,
    yani ``1 - cutoff`` guven seviyesinde periyot kaybinin %3'u
    gecmesi beklenmez). ``cutoff`` (0, 1) acik araliginda olmalidir.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff (0, 1) acik araliginda olmalidir")
    return float(np.percentile(r.to_numpy(), cutoff * 100))


def conditional_value_at_risk(returns, cutoff: float = 0.05) -> float:
    """Conditional VaR (CVaR / Expected Shortfall): VaR esiginin ALTINDA
    (esik dahil) kalan getirilerin ortalamasi.

    Once :func:`value_at_risk` ile esik (``cutoff`` persentili) bulunur,
    ardindan ``r <= VaR`` kosulunu saglayan ("kuyruktaki") getirilerin
    ortalamasi alinir. VaR gibi genellikle NEGATIFTIR ve buyuklukce
    ``|CVaR| >= |VaR|`` olur (kuyruktaki ortalama kayip, esik kaybindan
    en az o kadar buyuktur).
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    var = value_at_risk(r, cutoff)
    if np.isnan(var):
        return float("nan")
    tail = r[r <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())


def capture_ratio(returns, benchmark_returns, side: str = "up") -> float:
    """Yukari/asagi yakalama orani (upside/downside capture ratio).

    ``side="up"``: benchmark'in POZITIF oldugu donemlerde, stratejinin
    ortalama getirisi / benchmark'in ortalama getirisi.

    ``side="down"``: benchmark'in NEGATIF oldugu donemlerde ayni oran.

    "up" icin >1, stratejinin yukselislerde benchmark'tan daha fazla
    kazandigini; "down" icin (iki taraf da negatif oldugundan oran
    pozitif cikar) <1, stratejinin dususlerde benchmark'tan daha AZ
    kaybettigini gosterir. Getiriler ortak index uzerinde (ic kesisim)
    hizalanir; ilgili alt kumede benchmark getirisi bos/ortalamasi 0 ise
    NaN dondurur.
    """
    if side not in ("up", "down"):
        raise ValueError('side "up" ya da "down" olmalidir')
    aligned = _align_pair(returns, benchmark_returns)
    if len(aligned) == 0:
        return float("nan")
    mask = aligned["b"] > 0 if side == "up" else aligned["b"] < 0
    subset = aligned[mask]
    if len(subset) == 0:
        return float("nan")
    bench_mean = subset["b"].mean()
    if bench_mean == 0:
        return float("nan")
    return float(subset["r"].mean() / bench_mean)


def avg_win(returns) -> float:
    """Pozitif (kazandiran, ``r > 0``) getirilerin ortalamasi. Kazanc
    yoksa NaN dondurur."""
    r = _as_series(returns)
    wins = r[r > 0]
    if len(wins) == 0:
        return float("nan")
    return float(wins.mean())


def avg_loss(returns) -> float:
    """Negatif (kaybettiren, ``r < 0``) getirilerin ortalamasi. Kayip
    yoksa NaN dondurur.

    Donus degeri NEGATIFTIR (orn. -0.02).
    """
    r = _as_series(returns)
    losses = r[r < 0]
    if len(losses) == 0:
        return float("nan")
    return float(losses.mean())


def payoff_ratio(returns) -> float:
    """Payoff ratio = avg_win / |avg_loss|.

    Ortalama kazancin ortalama kayba (buyukluk olarak) oranidir; bkz.
    :func:`avg_win`, :func:`avg_loss`. Kayip yoksa (avg_loss NaN/0) ya
    da kazanc yoksa (avg_win NaN) NaN dondurur.
    """
    r = _as_series(returns)
    win = avg_win(r)
    loss = avg_loss(r)
    if np.isnan(win) or np.isnan(loss) or loss == 0:
        return float("nan")
    return float(win / abs(loss))


def kelly_criterion(returns) -> float:
    """Basit Kelly kriteri (f*): log-buyumeyi (uzun donem bilesik
    getiriyi) maksimize eden tahmini optimal bahis/kaldirac oranini
    verir.

    ``win_rate = pozitif getiri orani`` (bkz. :func:`hit_rate`)
    ``win_loss_ratio = avg_win / |avg_loss|`` (bkz. :func:`payoff_ratio`)

    ``f* = win_rate - (1 - win_rate) / win_loss_ratio``

    ``win_loss_ratio`` hesaplanamazsa (kayip ya da kazanc yoksa) ya da
    0 ise NaN dondurur. Negatif sonuc, stratejiye hic bahis yapmamanin
    (ya da ters pozisyonun) daha iyi oldugunu isaret eder.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    win_rate = hit_rate(r)
    win_loss_ratio = payoff_ratio(r)
    if np.isnan(win_rate) or np.isnan(win_loss_ratio) or win_loss_ratio == 0:
        return float("nan")
    return float(win_rate - (1.0 - win_rate) / win_loss_ratio)


def risk_of_ruin(returns, risk_fraction: float = 1.0) -> float:
    """Basitlestirilmis "risk of ruin" (sermayenin tukenme olasiligi)
    tahmini.

    VARSAYIMLAR (klasik trading risk yonetimi literaturunde yaygin
    kullanilan basit/kaba formul; getiri BUYUKLUKLERINI degil, sadece
    KAZANMA ORANINI (win_rate) kullanir):

    - "Edge" (kenar/avantaj), kazanma oraninin yazi-tura (%50)
      durumundan sapmasi olarak tanimlanir::

          edge = 2 * win_rate - 1      (win_rate > 0.5 ise pozitif)

    - ``risk_fraction``, her "bahis/islem"de riske atilan sermaye
      oranidir (orn. 0.02 = sermayenin %2'si). Sermayenin kac "birim"e
      bolundugu ``units = 1 / risk_fraction`` ile bulunur (orn.
      risk_fraction=0.02 -> 50 birim; yani 50 ardisik kayip sermayeyi
      tuketir).

    - Klasik risk-of-ruin yaklasimi (edge > 0 icin)::

          RoR = ((1 - edge) / (1 + edge)) ** units

      ``edge <= 0`` (kazanma sansi <= %50) ise strateji matematiksel
      beklentiyle uzun donemde sermayeyi tuketir ve RoR = 1.0
      (tukenme pratikte kesin) kabul edilir. ``edge >= 1`` (win_rate=1,
      hic kayip yok) durumunda RoR = 0.0 dondurulur.

    Bu, getiri BUYUKLUGUNU/oynakligini hesaba katan DETAYLI risk-of-ruin
    modellerinin (orn. Kelly-tabanli Monte Carlo simulasyonu) KABA bir
    yaklasimidir; kesin bir olasilik hesabi degildir. ``risk_fraction``
    0'dan buyuk olmalidir.
    """
    r = _as_series(returns)
    if len(r) == 0 or risk_fraction <= 0:
        return float("nan")
    win_rate = hit_rate(r)
    if np.isnan(win_rate):
        return float("nan")
    edge = 2.0 * win_rate - 1.0
    if edge <= 0:
        return 1.0
    if edge >= 1:
        return 0.0
    units = 1.0 / risk_fraction
    return float(((1.0 - edge) / (1.0 + edge)) ** units)


def ulcer_index(cumulative_returns) -> float:
    """Ulcer Index: drawdown serisinin RMS'i (karelerin ortalamasinin
    karekoku) - hem drawdown'un DERINLIGINI hem de SURESINI (uzun sure
    dusukte kalma) birlikte cezalandiran bir risk olcusudur.

    ``drawdown_t = curve_t / running_max_t - 1``  (bkz. :func:`max_drawdown`)

    ``UI = sqrt( mean(drawdown_t^2) )``

    Donus degeri daima >= 0'dir (buyuklugu drawdown birimindedir, orn.
    0.05 kabaca %5'lik "tipik" bir drawdown buyuklugune isaret eder).
    Bos seri icin NaN dondurur.
    """
    curve = _as_series(cumulative_returns)
    if len(curve) == 0:
        return float("nan")
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(np.sqrt(np.mean(drawdown.to_numpy() ** 2)))


def common_sense_ratio(returns) -> float:
    """Common Sense Ratio (QuantStats tanimi): Tail Ratio * Profit Factor.

    ``CSR = tail_ratio(r) * profit_factor(r)``

    bkz. :func:`tail_ratio`, :func:`profit_factor`. Hem getiri
    dagiliminin kuyruk (ekstrem deger) karakteristigini hem de genel
    kazanc/kayip karakteristigini tek bir sayida birlestirir.
    """
    r = _as_series(returns)
    return float(tail_ratio(r) * profit_factor(r))


def recovery_factor(cumulative_returns) -> float:
    """Recovery Factor = Toplam getiri / |Maksimum drawdown|.

    ``toplam_getiri = curve[-1] / curve[0] - 1``  (kumulatif buyume
    egrisinin bas ve sonu arasindaki toplam getiri)

    ``Recovery Factor = toplam_getiri / |max_drawdown|``

    bkz. :func:`max_drawdown`. En az 2 gozlem gerekir; max_drawdown
    0 ya da NaN ise NaN dondurur.
    """
    curve = _as_series(cumulative_returns)
    if len(curve) < 2:
        return float("nan")
    mdd = max_drawdown(curve)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    total_return = curve.iloc[-1] / curve.iloc[0] - 1.0
    return float(total_return / abs(mdd))


def gain_to_pain_ratio(returns) -> float:
    """Gain-to-Pain Ratio = Toplam getiri / Toplam |negatif getiriler|.

    ``toplam_getiri = sum(r)``  (BASIT/aritmetik toplam, compound DEGIL)
    ``toplam_kayip = |sum(r[r < 0])|``

    Profit factor'a benzer, ancak payda sadece kayiplarin toplamini
    kullanirken pay TUM getirilerin (kazanc - kayip) toplamini kullanir
    (profit_factor'da pay sadece kazanclardir). Kayip yoksa ve pay > 0
    ise +inf, hem pay hem kayip 0 ise NaN dondurur.
    """
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    total_return = r.sum()
    total_loss = abs(r[r < 0].sum())
    if total_loss == 0:
        return float("inf") if total_return > 0 else float("nan")
    return float(total_return / total_loss)
