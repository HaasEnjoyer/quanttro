# `quanttro.propfirm`

## Scope — read this before anything else

This module does not touch real money, place orders, or integrate with any
prop-firm's systems. It answers one narrow, purely mathematical question:
*"if this return series had been run under rule set X (daily loss limit,
max drawdown, profit target, consistency rule), would it have passed, and
on which day would it have failed if not?"* — the same category of analysis
as `quanttro.backtest.max_drawdown`, just interpreted against a specific
rule set instead of a generic threshold.

`FTMO_CHALLENGE_RULES` and `TOPSTEP_COMBINE_RULES` are **illustrative
example presets** built from each firm's generally published rule
structure — exact numeric thresholds change over time, so verify against
the firm's current official rules page before drawing any real conclusion.
This module is a research/estimation aid, not an approval or guarantee.

**Trademark / non-affiliation notice:** "FTMO" and "Topstep" are trademarks
of their respective owners. This project is not affiliated with, endorsed
by, or sponsored by either company — the names are used only to describe
which firm's generally published rule *structure* a preset approximates
(nominative/descriptive use), not to claim an official or verified replica
of their current terms.

## Why this exists

Retail "funded trader" evaluation challenges are a large, real market
(prop-firm evaluation-fee revenue has been estimated in the hundreds of
millions of dollars per year), and most publicly available "will I pass"
tools use synthetic Monte Carlo simulation from an assumed win-rate — not
an actual historical equity curve run against a firm's specific rule
mechanics (daily-loss-limit reset logic, static vs. trailing drawdown,
consistency rules). If you already have a backtested strategy's daily
returns, this module answers that question directly against real history
instead of a synthetic assumption.

This isn't a hypothetical gap. Reading real user reviews of prop firms
(Trustpilot pages for FTMO, Topstep, The5ers, Apex Trader Funding) turns up
a recurring, specific complaint: traders breached and had their accounts
closed, and support could not or would not name the exact trade/timestamp
that triggered the breach — the enforcement is opaque from the trader's
side. A tool that lets a trader independently replay their own history
against the firm's *stated* rules, before relying on the firm's own
dashboard, addresses exactly that trust gap. (A paid competitor,
TradesViz's "Retroactive Evaluation" feature, charges $19.99–$29.99/month
for a similar compliance-replay check across ~20 prop firms — concrete
evidence this is a feature people already pay for, not a hypothetical one.)

## `PropFirmRules`

```python
from quanttro.propfirm import PropFirmRules

custom_rules = PropFirmRules(
    profit_target_pct=0.08,
    max_daily_loss_pct=0.04,
    max_overall_loss_pct=0.08,
    drawdown_type="trailing",  # "static" or "trailing"
    min_trading_days=5,
    consistency_rule_pct=0.30,  # optional, simplified single-day-dominance check
)
```

- `drawdown_type="static"`: the max-loss threshold is always measured
  against the **starting** balance.
- `drawdown_type="trailing"`: the threshold is measured against the
  **highest balance ever reached** — stricter, since the bar rises as the
  account grows. This distinction matters: a return series can pass under
  static rules and fail under trailing rules on identical data (demonstrated
  below).
- `max_daily_loss_pct=None`: some evaluation programs don't enforce a daily
  loss limit at all — only an overall (often trailing) drawdown limit. Pass
  `None` rather than a very large number to disable the daily check
  entirely; see `TRAILING_ONLY_NO_DAILY_LIMIT_RULES` for a generic (not
  tied to a named firm) preset illustrating this structure.

## `simulate_prop_firm_challenge`

All examples below were actually run against this repo's venv.

```python
import pandas as pd
from quanttro.propfirm import FTMO_CHALLENGE_RULES, simulate_prop_firm_challenge

idx = pd.date_range("2026-01-01", periods=10, freq="D")
daily_returns = pd.Series([0.015] * 10, index=idx)

result = simulate_prop_firm_challenge(daily_returns, FTMO_CHALLENGE_RULES, starting_balance=100_000)
print(result.passed, result.outcome, result.pass_day, result.trading_days_used)
```

**Output:**
```
True passed 2026-01-07 00:00:00 7
```

The challenge would have passed on day 7 (profit target reached, and
`min_trading_days=4` was already satisfied).

### Static vs. trailing drawdown — same data, different outcome

```python
from quanttro.propfirm import PropFirmRules, TOPSTEP_COMBINE_RULES

idx = pd.date_range("2026-01-01", periods=5, freq="D")
rets = pd.Series([0.01, 0.02, -0.015, -0.015, -0.015], index=idx)

trailing = simulate_prop_firm_challenge(rets, TOPSTEP_COMBINE_RULES, starting_balance=50_000)
print(trailing.passed, trailing.outcome)

static_rules = PropFirmRules(
    profit_target_pct=0.06, max_daily_loss_pct=0.02,
    max_overall_loss_pct=0.04, drawdown_type="static", min_trading_days=2,
)
static = simulate_prop_firm_challenge(rets, static_rules, starting_balance=50_000)
print(static.passed, static.outcome)
```

**Output:**
```
False max_drawdown_breached
False incomplete
```

The account rises to a peak of ~51,510 then pulls back three days in a row.
Measured from that peak (trailing), the pullback exceeds the 4% limit and
the challenge fails on day 5. Measured from the starting balance (static),
the same pullback never exceeds 4% from the original 50,000 — the challenge
doesn't fail, it just runs out of days without reaching the profit target
either (`outcome="incomplete"`). This is exactly why `drawdown_type`
matters when picking which firm's rules to simulate against.

### Reading `outcome`

| `outcome` | Meaning |
|---|---|
| `"passed"` | Profit target reached, `min_trading_days` satisfied, consistency rule (if any) not violated. |
| `"daily_loss_limit_breached"` | A single day's loss (vs. the previous day's closing balance) exceeded `max_daily_loss_pct`. |
| `"max_drawdown_breached"` | Cumulative loss (vs. start or peak, per `drawdown_type`) exceeded `max_overall_loss_pct`. |
| `"consistency_rule_violated"` | Target was reached, but one day's profit exceeded `consistency_rule_pct` of the total — a simplified approximation of real consistency rules, which can be more involved. |
| `"incomplete"` | The series ended without passing or triggering any failure condition — inconclusive, not a pass. |

### No-daily-limit programs

```python
from quanttro.propfirm import TRAILING_ONLY_NO_DAILY_LIMIT_RULES

idx = pd.date_range("2026-01-01", periods=5, freq="D")
rets = pd.Series([0.05, 0.05, -0.08, 0.02, 0.02], index=idx)

result = simulate_prop_firm_challenge(rets, TRAILING_ONLY_NO_DAILY_LIMIT_RULES, starting_balance=100_000)
print(result.outcome)
```

**Output:**
```
max_drawdown_breached
```

A single day's -8% loss would trip most daily-loss limits instantly, but
this rule set has `max_daily_loss_pct=None` — it's never checked. The
challenge still fails, but on the overall trailing-drawdown rule instead,
confirming the daily check is genuinely skipped rather than silently
passing.
