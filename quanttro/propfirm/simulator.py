"""
quanttro.propfirm.simulator
================================
Bir getiri serisini :class:`PropFirmRules` kurallarina gore gun-gun tarayan
simulasyon fonksiyonu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .rules import PropFirmRules

_OUTCOME_PASSED = "passed"
_OUTCOME_DAILY_LOSS = "daily_loss_limit_breached"
_OUTCOME_MAX_DRAWDOWN = "max_drawdown_breached"
_OUTCOME_CONSISTENCY = "consistency_rule_violated"
_OUTCOME_INCOMPLETE = "incomplete"


@dataclass
class PropFirmResult:
    """:func:`simulate_prop_firm_challenge` sonucu."""

    passed: bool
    outcome: str
    fail_day: pd.Timestamp | None
    pass_day: pd.Timestamp | None
    trading_days_used: int
    final_balance: float
    max_drawdown_seen: float
    daily_balances: list[float] = field(default_factory=list)


def simulate_prop_firm_challenge(
    daily_returns: pd.Series,
    rules: PropFirmRules,
    starting_balance: float = 100_000.0,
) -> PropFirmResult:
    """
    ``daily_returns`` (gunluk yuzdesel getiri serisi, orn.
    ``equity.pct_change()``) serisini ``rules`` kurallarina gore GUN GUN
    tarar ve sinavin GECERDI mi, GECMEZDI mi, hangi gun hangi kuraldan
    basarisiz OLURDU sorusunu cevaplar.

    Tarama mantigi her gun icin:

    1. Getiri bakiyeye uygulanir.
    2. GUNLUK KAYIP kontrolu: bakiye, O GUNUN BASINDAKI bakiyeye (onceki
       gunun kapanisi) gore ``rules.max_daily_loss_pct``'tan fazla
       dusmusse -> ``daily_loss_limit_breached`` ile BASARISIZ.
    3. MAKSIMUM DRAWDOWN kontrolu: ``rules.drawdown_type`` "static" ise
       baslangic bakiyesine, "trailing" ise o ana kadar gorulen en yuksek
       bakiyeye (peak) gore dusus ``rules.max_overall_loss_pct``'i
       asmissa -> ``max_drawdown_breached`` ile BASARISIZ.
    4. KAR HEDEFI kontrolu: toplam kar ``rules.profit_target_pct``'e
       ulasmis VE (varsa) ``rules.min_trading_days`` kadar gun gecmisse ->
       (varsa) tutarlilik kurali kontrol edilip BASARILI sayilir.

    Parametreler
    ------------
    daily_returns : pd.Series
        Gunluk yuzdesel getiri serisi (orn. 0.01 = %1 kar). Index tarih
        olmalidir (raporlamada ``fail_day``/``pass_day`` icin kullanilir).
    rules : PropFirmRules
        Uygulanacak kural kumesi.
    starting_balance : float
        Degerlendirme hesabinin baslangic bakiyesi.

    Donus
    -----
    PropFirmResult
        ``outcome`` dortu "passed" | "daily_loss_limit_breached" |
        "max_drawdown_breached" | "consistency_rule_violated" |
        "incomplete" (seri bitti, ne gecti ne basarisiz oldu) degerlerinden
        biridir.

    Hatalar
    -------
    ValueError
        ``daily_returns`` bossa.
    """
    if len(daily_returns) == 0:
        raise ValueError("daily_returns bos olamaz")

    balance = float(starting_balance)
    day_start_balance = balance
    peak_balance = balance
    daily_balances: list[float] = []
    max_drawdown_seen = 0.0

    daily_pnls: list[float] = []

    for day_index, (idx, ret) in enumerate(daily_returns.items(), start=1):
        balance *= 1.0 + float(ret)
        daily_balances.append(balance)
        daily_pnls.append(balance - day_start_balance)

        daily_loss_pct = (balance - day_start_balance) / day_start_balance
        if rules.max_daily_loss_pct is not None and daily_loss_pct <= -rules.max_daily_loss_pct:
            return PropFirmResult(
                passed=False,
                outcome=_OUTCOME_DAILY_LOSS,
                fail_day=idx,
                pass_day=None,
                trading_days_used=day_index,
                final_balance=balance,
                max_drawdown_seen=max_drawdown_seen,
                daily_balances=daily_balances,
            )

        reference = starting_balance if rules.drawdown_type == "static" else peak_balance
        drawdown_pct = (balance - reference) / reference
        max_drawdown_seen = min(max_drawdown_seen, drawdown_pct)
        if drawdown_pct <= -rules.max_overall_loss_pct:
            return PropFirmResult(
                passed=False,
                outcome=_OUTCOME_MAX_DRAWDOWN,
                fail_day=idx,
                pass_day=None,
                trading_days_used=day_index,
                final_balance=balance,
                max_drawdown_seen=max_drawdown_seen,
                daily_balances=daily_balances,
            )

        peak_balance = max(peak_balance, balance)

        total_profit_pct = (balance - starting_balance) / starting_balance
        days_ok = rules.min_trading_days is None or day_index >= rules.min_trading_days
        if total_profit_pct >= rules.profit_target_pct and days_ok:
            total_profit = balance - starting_balance
            if rules.consistency_rule_pct is not None and total_profit > 0:
                best_day_pnl = max(daily_pnls)
                if best_day_pnl / total_profit > rules.consistency_rule_pct:
                    return PropFirmResult(
                        passed=False,
                        outcome=_OUTCOME_CONSISTENCY,
                        fail_day=idx,
                        pass_day=None,
                        trading_days_used=day_index,
                        final_balance=balance,
                        max_drawdown_seen=max_drawdown_seen,
                        daily_balances=daily_balances,
                    )
            return PropFirmResult(
                passed=True,
                outcome=_OUTCOME_PASSED,
                fail_day=None,
                pass_day=idx,
                trading_days_used=day_index,
                final_balance=balance,
                max_drawdown_seen=max_drawdown_seen,
                daily_balances=daily_balances,
            )

        day_start_balance = balance

    return PropFirmResult(
        passed=False,
        outcome=_OUTCOME_INCOMPLETE,
        fail_day=None,
        pass_day=None,
        trading_days_used=len(daily_returns),
        final_balance=balance,
        max_drawdown_seen=max_drawdown_seen,
        daily_balances=daily_balances,
    )
