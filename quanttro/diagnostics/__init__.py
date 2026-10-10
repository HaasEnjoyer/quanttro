"""quanttro.diagnostics - BIST'e ozgu veri/piyasa anomalilerini otomatik tespit eder.

Bu paket, elle yapilan kesif analizlerinde (KONTR'da tavan-serisi anomalisi, haftalik
hacim-getiri iliskisinde rejim kirilmasi gibi) bulunan paternleri tekrar kullanilabilir
fonksiyonlara donusturur:

- `anomalies`: tavan/taban (limit-up/limit-down) serilerini ve coklu hisseler
  arasindaki aykiri (outlier) davranisi tespit eder.
- `regime`: walk-forward analizlerde rejim kirilmasini (isaret degisimi) ve iki
  serinin korelasyon stabilitesini olcer.

Detaylar icin ilgili modulun docstring'lerine bakin. Metodolojik tuzaklar (nelere
dikkat edilmeli, hangi varsayimlar kirilabilir) icin bkz. docs/KNOWN_PITFALLS.md.
"""

from quanttro.diagnostics.anomalies import (
    detect_limit_streaks,
    flag_outlier_stocks,
    limit_frequency_by_year,
)
from quanttro.diagnostics.regime import (
    detect_regime_break,
    rolling_correlation_stability,
)

__all__ = [
    "detect_limit_streaks",
    "limit_frequency_by_year",
    "flag_outlier_stocks",
    "detect_regime_break",
    "rolling_correlation_stability",
]
