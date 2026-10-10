"""
quanttro.propfirm
======================
Bir stratejinin getiri serisini, prop-firm ("funded trader") degerlendirme
sinavlarinin KURALLARINA gore GECMISE DONUK simule eden modul.

ONEMLI - KAPSAM: Bu modul hicbir gercek paraya, emre, araciliga ya da bir
prop-firm ile entegrasyona DOKUNMAZ. Yaptigi sey tamamen matematikseldir:
kullanicinin elindeki bir getiri serisini alir, verilen kural kumesine
(gunluk kayip limiti, maksimum drawdown - statik ya da trailing - kar
hedefi, tutarlilik/consistency kurali) gore gun gun tarar ve "bu seri bu
kurallarla calistirilsaydi sinavi GECER miydi, hangi gun ihlal olurdu"
sorusunu cevaplar. Bu, modulun diger metrik fonksiyonlarindan
(:func:`quanttro.backtest.max_drawdown` gibi) farkli bir hukuki kategori
DEGILDIR - sadece ozel bir kural kumesine gore yorumlanmis bir drawdown/
getiri analizidir.

DIKKAT: :data:`FTMO_CHALLENGE_RULES` ve :data:`TOPSTEP_COMBINE_RULES` gibi
hazir kural setleri ORNEKTIR ve gercek firmalarin KENDI YAYINLADIGI genel
kural yapisina dayanir - prop-firm'lerin tam sayisal esikleri SIK SIK
DEGISIR. Gercek bir degerlendirmeye girmeden once MUTLAKA ilgili firmanin
o anki guncel, resmi kural sayfasindan dogrulayin; bu modul bir onay/
garanti ARACI DEGILDIR, sadece bir arastirma/tahmin yardimcisidir.

TICARI MARKA / BAGLANTISIZLIK: "FTMO" ve "Topstep" ilgili sahiplerinin
ticari markalaridir. Bu proje bu firmalarla BAGLANTILI DEGILDIR, onlar
tarafindan ONAYLANMAMIS ya da SPONSOR OLUNMAMISTIR; isimler sadece hangi
firmanin GENEL YAYINLADIGI kural YAPISINA atifta bulunuldugunu belirtmek
icin (tanimlayici/nominative kullanim) gecer. Bkz. ``rules.py`` modul
dokstring'i icin tam ibare.

Alt parcalar
------------
- :class:`PropFirmRules`: bir degerlendirme sinavinin kural kumesi.
- :data:`FTMO_CHALLENGE_RULES`, :data:`TOPSTEP_COMBINE_RULES`: ornek
  (ve DOGRULANMASI gereken) hazir kural setleri.
- :func:`simulate_prop_firm_challenge`: asil simulasyon fonksiyonu.
- :class:`PropFirmResult`: simulasyon sonucu.
"""

from __future__ import annotations

from .rules import (
    FTMO_CHALLENGE_RULES,
    TOPSTEP_COMBINE_RULES,
    TRAILING_ONLY_NO_DAILY_LIMIT_RULES,
    PropFirmRules,
)
from .simulator import PropFirmResult, simulate_prop_firm_challenge

__all__ = [
    "PropFirmRules",
    "FTMO_CHALLENGE_RULES",
    "TOPSTEP_COMBINE_RULES",
    "TRAILING_ONLY_NO_DAILY_LIMIT_RULES",
    "PropFirmResult",
    "simulate_prop_firm_challenge",
]
