"""quanttro.reports - Bir sembol/analiz icin otomatik ozet rapor uretir.

Iki rapor motoru sunuyor:

- `generate_symbol_report` (bkz. `generate.py`): fiyat/hacim ozet
  istatistikleri ile `quanttro.diagnostics.anomalies` sonuclarini
  birlestirip basit bir Markdown ya da HTML rapor dosyasi uretir.
- `tearsheet` (bkz. `tearsheet.py`): QuantStats tarzi, getiri serisi
  uzerinden kapsamli bir performans karnesi (Sharpe/Sortino/Calmar,
  aylik getiri isi haritasi, drawdown donemleri, opsiyonel olarak
  benchmark karsilastirmasi) uretir. `monthly_returns_table`,
  `rolling_sharpe` ve `drawdown_periods` da bagimsiz olarak kullanilabilir.
"""

from quanttro.reports.generate import generate_symbol_report
from quanttro.reports.tearsheet import (
    drawdown_periods,
    monthly_returns_table,
    rolling_sharpe,
    tearsheet,
)

__all__ = [
    "generate_symbol_report",
    "tearsheet",
    "monthly_returns_table",
    "rolling_sharpe",
    "drawdown_periods",
]
