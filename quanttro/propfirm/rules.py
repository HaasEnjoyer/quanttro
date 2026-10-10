"""
quanttro.propfirm.rules
============================
Prop-firm degerlendirme sinavlarinin kural temsili.

TICARI MARKA / BAGLANTISIZLIK UYARISI: Bu dosyadaki FTMO(R), Topstep(R),
The5ers(R) ve Apex Trader Funding(R) isimleri KENDI SAHIPLERININ ticari
markalaridir. Bu proje bu firmalardan HICBIRIYLE BAGLANTILI, ONLAR
TARAFINDAN ONAYLANMIS ya da SPONSOR OLUNMUS DEGILDIR. Isimler SADECE hangi
firmanin GENEL YAYINLADIGI kural YAPISINA atifta bulunuldugunu belirtmek
icin (nominative/tanimlayici kullanim) kullanilmistir - asagidaki sayisal
esikler o firmalarin RESMI, GUNCEL kurallarinin BIREBIR KOPYASI DEGILDIR,
YAKLASIK bir temsildir ve ZAMANLA DEGISEBILIR. Gercek bir degerlendirmeye
girmeden once MUTLAKA ilgili firmanin kendi resmi/guncel kural sayfasindan
dogrulayin - bu modul bir onay/garanti araci degildir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PropFirmRules:
    """
    Bir prop-firm degerlendirme (evaluation/challenge) sinavinin kural
    kumesi.

    Parametreler
    ------------
    profit_target_pct : float
        Sinavi gecmek icin ulasilmasi gereken kar hedefi (orn. 0.10 = %10).
    max_daily_loss_pct : float | None
        Bir gun icinde (onceki gun sonu bakiyesine gore) izin verilen azami
        kayip (orn. 0.05 = %5). Bu esik asilirsa sinav O GUN basarisiz olur.
        ``None`` ise GUNLUK kayip kontrolu hic YAPILMAZ - bazi firmalar
        (orn. sadece toplam/trailing drawdown kurali olan, gunluk limiti
        olmayan programlar) icin bu kontrolun tamamen devre disi birakilmasi
        gerekir; bunu rasgele buyuk bir sayiyla (orn. 1.0) "etkisiz hale
        getirmeye" CALISMAYIN, dogrudan ``None`` kullanin.
    max_overall_loss_pct : float
        Toplamda izin verilen azami kayip (orn. 0.10 = %10).
        ``drawdown_type``'a gore baslangic bakiyesine ya da en yuksek
        gorulen bakiyeye (peak) gore olculur.
    drawdown_type : "static" | "trailing"
        "static": kayip her zaman BASLANGIC bakiyesine gore olculur (bakiye
        yukseldikce esik sabit kalir). "trailing": kayip EN YUKSEK GORULEN
        bakiyeye (peak) gore olculur (esik bakiye yukseldikce birlikte
        yukselir) - bu, "static"e gore DAHA SIKI bir kuraldir.
    min_trading_days : int | None
        Kar hedefine ulasilsa bile sinavin gecerli sayilmasi icin gereken
        azami islem/gun sayisi. ``None`` ise kontrol edilmez.
    consistency_rule_pct : float | None
        BASITLESTIRILMIS tutarlilik kurali: toplam kardaki TEK BIR gunun
        payi bu orani asarsa (orn. 0.30 = tek gun toplam karin %30'undan
        fazlasini olusturuyorsa) sinav basarisiz sayilir. DIKKAT: gercek
        prop-firm'lerin tutarlilik kurallari genelde bundan daha karmasiktir
        (haftalik dagilim, islem sayisi gibi ek kriterler icerebilir) - bu
        alan sadece en yaygin basit varyanti yaklasik olarak modeller.
        ``None`` ise kontrol edilmez.
    name : str
        Kural setinin adi (raporlama/okunabilirlik icin).
    """

    profit_target_pct: float
    max_daily_loss_pct: float | None
    max_overall_loss_pct: float
    drawdown_type: Literal["static", "trailing"] = "static"
    min_trading_days: int | None = None
    consistency_rule_pct: float | None = None
    name: str = "custom"


# ---------------------------------------------------------------------------
# ORNEK kural setleri - bkz. modul dokstring'indeki DIKKAT notu: bunlar
# firmalarin GENEL YAYINLADIGI kural YAPISINA dayanan ORNEKLERDIR, tam
# sayisal esikler zaman icinde degisebilir. Gercek kullanimdan once ilgili
# firmanin GUNCEL kural sayfasindan dogrulayin.
# ---------------------------------------------------------------------------

FTMO_CHALLENGE_RULES = PropFirmRules(
    profit_target_pct=0.10,
    max_daily_loss_pct=0.05,
    max_overall_loss_pct=0.10,
    drawdown_type="static",
    min_trading_days=4,
    name="FTMO Challenge (ornek/dogrulanmasi gereken)",
)

TOPSTEP_COMBINE_RULES = PropFirmRules(
    profit_target_pct=0.06,
    max_daily_loss_pct=0.02,
    max_overall_loss_pct=0.04,
    drawdown_type="trailing",
    min_trading_days=2,
    name="Topstep Trading Combine (ornek/dogrulanmasi gereken)",
)

# Bazi futures degerlendirme programlari GUNLUK kayip limiti UYGULAMAZ,
# sadece toplam (trailing) drawdown'a bakar. Belirli bir firmayi
# isimlendirmek yerine bu YAPIYI jenerik olarak temsil ediyoruz - gercek bir
# firmanin kurallarini modellemek icin PropFirmRules'u kendi guncel kural
# sayfasindan dogrulayarak OLUSTURUN.
TRAILING_ONLY_NO_DAILY_LIMIT_RULES = PropFirmRules(
    profit_target_pct=0.08,
    max_daily_loss_pct=None,
    max_overall_loss_pct=0.06,
    drawdown_type="trailing",
    min_trading_days=7,
    name="Jenerik: sadece trailing drawdown, gunluk limit yok (ornek yapi)",
)
