"""quanttro.propfirm paketi icin birim testleri.

pytest ile calistirilir::

    python -m pytest tests/test_propfirm.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from quanttro.propfirm.rules import (
    FTMO_CHALLENGE_RULES,
    TOPSTEP_COMBINE_RULES,
    TRAILING_ONLY_NO_DAILY_LIMIT_RULES,
    PropFirmRules,
)
from quanttro.propfirm.simulator import simulate_prop_firm_challenge


def make_returns(values, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_passes_when_target_reached_without_violations():
    rets = make_returns([0.015] * 10)
    result = simulate_prop_firm_challenge(rets, FTMO_CHALLENGE_RULES, starting_balance=100_000)
    assert result.passed
    assert result.outcome == "passed"
    assert result.trading_days_used == 7
    assert result.final_balance > 100_000


def test_fails_on_daily_loss_limit_breach():
    rets = make_returns([0.01, 0.01, -0.06, 0.01])
    result = simulate_prop_firm_challenge(rets, FTMO_CHALLENGE_RULES, starting_balance=100_000)
    assert not result.passed
    assert result.outcome == "daily_loss_limit_breached"
    assert result.trading_days_used == 3


def test_trailing_drawdown_fails_where_static_would_not():
    rets = make_returns([0.01, 0.02, -0.015, -0.015, -0.015])

    trailing_result = simulate_prop_firm_challenge(rets, TOPSTEP_COMBINE_RULES, starting_balance=50_000)
    assert not trailing_result.passed
    assert trailing_result.outcome == "max_drawdown_breached"

    static_rules = PropFirmRules(
        profit_target_pct=0.06,
        max_daily_loss_pct=0.02,
        max_overall_loss_pct=0.04,
        drawdown_type="static",
        min_trading_days=2,
    )
    static_result = simulate_prop_firm_challenge(rets, static_rules, starting_balance=50_000)
    assert static_result.outcome != "max_drawdown_breached"


def test_consistency_rule_rejects_single_day_dominated_profit():
    rules = PropFirmRules(
        profit_target_pct=0.10,
        max_daily_loss_pct=0.05,
        max_overall_loss_pct=0.10,
        consistency_rule_pct=0.3,
        min_trading_days=2,
    )
    rets = make_returns([0.09, 0.005, 0.01])
    result = simulate_prop_firm_challenge(rets, rules, starting_balance=100_000)
    assert not result.passed
    assert result.outcome == "consistency_rule_violated"


def test_consistency_rule_allows_evenly_distributed_profit():
    rules = PropFirmRules(
        profit_target_pct=0.10,
        max_daily_loss_pct=0.05,
        max_overall_loss_pct=0.10,
        consistency_rule_pct=0.3,
        min_trading_days=2,
    )
    rets = make_returns([0.025] * 5)
    result = simulate_prop_firm_challenge(rets, rules, starting_balance=100_000)
    assert result.passed
    assert result.outcome == "passed"


def test_incomplete_when_series_ends_without_pass_or_fail():
    rets = make_returns([0.001] * 5)
    result = simulate_prop_firm_challenge(rets, FTMO_CHALLENGE_RULES, starting_balance=100_000)
    assert not result.passed
    assert result.outcome == "incomplete"
    assert result.trading_days_used == 5


def test_min_trading_days_delays_pass_even_if_target_reached_early():
    rules = PropFirmRules(
        profit_target_pct=0.02,
        max_daily_loss_pct=0.05,
        max_overall_loss_pct=0.10,
        min_trading_days=3,
    )
    rets = make_returns([0.03, 0.001, 0.001])
    result = simulate_prop_firm_challenge(rets, rules, starting_balance=100_000)
    assert result.passed
    assert result.trading_days_used == 3


def test_rejects_empty_series():
    with pytest.raises(ValueError):
        simulate_prop_firm_challenge(pd.Series(dtype=float), FTMO_CHALLENGE_RULES)


def test_daily_balances_tracks_every_day():
    rets = make_returns([0.01, 0.01, 0.01])
    result = simulate_prop_firm_challenge(rets, FTMO_CHALLENGE_RULES, starting_balance=100_000)
    assert len(result.daily_balances) == 3


def test_none_daily_loss_limit_skips_daily_check_but_overall_still_applies():
    # Tek gunde -8% gibi buyuk bir kayip, gunluk limit kontrolu "None" ile
    # devre disi oldugu icin O ASAMADA basarisiz saymaz, ama toplam
    # (trailing) drawdown kontrolu hala calisir ve nihayetinde tetiklenir.
    rets = make_returns([0.05, 0.05, -0.08, 0.02, 0.02])
    result = simulate_prop_firm_challenge(
        rets, TRAILING_ONLY_NO_DAILY_LIMIT_RULES, starting_balance=100_000
    )
    assert result.outcome != "daily_loss_limit_breached"
    assert result.outcome == "max_drawdown_breached"
