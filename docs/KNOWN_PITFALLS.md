# Known Pitfalls

This document is a summary of the methodological lessons found manually during
exploratory analyses on `bist-backtest` / `quanttro` (October 2026). The goal is to
avoid repeating the same mistakes and to correctly interpret the results of
`quanttro.diagnostics`. Each item references the analysis script/investigation in
which the finding was first discovered.

---

## 1. Market-wide shock days must be winsorized

**Finding:** On market-wide (macro/news-driven) shock days such as March 19, 2025, the
large majority of BIST stocks show extreme returns simultaneously. This is not a
"stock-specific" behavior of a single stock, but an event affecting the entire market.

**Problem:** When working with non-winsorized raw returns, days like this:
- Dominate volatility/statistics calculations (mean, std, ACF),
- Distort learning in model training (LightGBM, etc.) by acting as overweighted samples,
- Can cause a single day to be misread as a "stock-specific signal."

**Solution:** As in `feature_engineering.py`, returns are winsorized with `RETURN_CLIP`
(e.g. 0.12) via `.clip(-RETURN_CLIP, RETURN_CLIP)`. This does NOT remove limit-up/limit-down
(tavan/taban, ~9.5%-10%) days (those are real events that need to be analyzed), it only
prevents extreme outliers from dominating the statistics. A similarly looser clip is
applied to weekly aggregations as well (see `analyze_weekly_volume.py` —
`RETURN_CLIP = 0.5`).

**Rule of thumb:** When adding a new feature/target, ask "how extreme a value could
this take on a market-wide shock day?" and apply winsorizing/clipping accordingly.

---

## 2. Macro features like USD/TRY can create spurious correlation with the inflation trend — detrending is required

**Finding:** USD/TRY is almost always trending upward due to Turkey's structural
inflation/devaluation trend. If the raw `usdtry_ret_1` is used, since this shared trend
is also an "always up" component shared with the raw returns of BIST stocks, the model
learns a **spurious/common-trend correlation**, not a genuine relationship.

**Solution (two parts):**
1. **On the target side:** instead of the raw forward return, the **relative (excess)
   return** against the BIST30 average is targeted (`target_fwd_rel_ret` =
   `target_fwd_ret - market_average`, see `feature_engineering.py::add_relative_target`).
   This removes the market-wide trend shared across all stocks (including TL
   devaluation) from the target.
2. **On the feature side:** the USD/TRY feature is also used as a **deviation from
   trend** instead of the raw return: `usdtry_ret_1_detrended = usdtry_ret_1 -
   usdtry_ret_1.rolling(TREND_WINDOW).mean()` (see
   `feature_engineering.py::load_macro_aligned`). This carries information about "how
   much of a surprise deviation from trend the current move represents," not the
   long-term devaluation trend itself.

**Rule of thumb:** When adding any macro series (USD/TRY, interest rates, inflation,
etc.) as a feature, ask "does this series carry a structural trend, and is that trend
shared with the target variable?" — if the answer is yes, detrend it.

---

## 3. Turning zero-volume bars into NaN when computing the volume z-score is wrong

**Finding:** When computing the volume z-score (`volume_zscore = (volume - rolling_mean) / rolling_std`),
it may be tempting to treat zero- or very-low-volume bars as "invalid" and convert them
to NaN, but this is an **incorrect** intervention.

**Problem:** A zero-volume bar (e.g. an hour/day with no trading, the first bar after a
holiday, or a moment that genuinely saw very little trading) is a **real market
condition**, not a data error. Turning these bars into NaN and excluding them from the
rolling window:
- Needlessly distorts the effective length of the rolling window (the window is no
  longer "the last N bars" but "the last N valid bars," making z-scores incomparable
  across different stocks/periods),
- Discards information — low volume itself can be an important signal (e.g. low
  participation before/after a holiday); erasing that is a loss of information.

**Solution:** Include zero-volume bars **directly** in the rolling computation
(`feature_engineering.py::build_features` — `volume.rolling(20).mean()` / `.std()`
operates directly on the raw `Volume` column, with no special NaN-masking).

**Rule of thumb:** Distinguish "is this value genuinely invalid (sensor/data error), or
just low/zero?" Only the former should be converted to NaN.

---

## 4. In low-float stocks, autocorrelation does not mean "predictability" — it can be a sign of manipulation/distress

**Finding:** In the scan done with `analyze_autocorrelation.py`, KONTR stands out
noticeably from the other BIST30 stocks in lag-1 return autocorrelation (~0.207); it
also has a limit-up/limit-down streak frequency (detected via `detect_limit_streaks`,
the longest being 8 days, 2020-10-20 → 2020-10-30) and yearly limit-up+limit-down
ratios (`limit_frequency_by_year`: ~27% in 2020, ~20% in 2026, see
`data/predictions/autocorrelation_summary.csv` and the KONTR research notes) far
higher than the others.

**Critical distinction:** This HIGH AUTOCORRELATION should NOT be read as "this stock
is predictable, it can be profited from with a trend-following strategy." Research
(see `kontr_limit_streak_research.md`, `kontr_bist_mechanics_research.md`) shows the
following:

- Literature on China/Taiwan limit-up/limit-down (tavan/taban) markets shows that
  1-day continuation after a limit hit is partly **mechanical** (the limit rule carries
  the day's real demand over to the next day), and that this turns into **reversal at
  a weekly horizon** — not "momentum that goes on forever."
- Low float combined with concentrated ownership is, as confirmed in the literature,
  the classic risk profile for both the "magnet effect" (microstructural) and
  **pump-and-dump/manipulation** — it is IMPOSSIBLE to distinguish the two from price
  data alone.
- KONTR's own concrete history confirms this: the limit-up streaks in the 2020-2022
  period (float ~20-29%) fit a "pump" profile, while the limit-down streaks in the
  2025-2026 period coincide with VBTS (Volatility-Based Trading Suspension) measures,
  being moved to the Close Monitoring Market ("Yakın İzleme Pazarı"), and a bond
  default — **the same statistical pattern (high autocorrelation/limit-streak
  frequency) can arise under two entirely different regimes (pump vs. distress).**

**Rule of thumb:** Do not automatically jump to the conclusion of "a tradeable signal"
for a stock flagged by functions like `flag_outlier_stocks`. First check the
free-float/ownership structure and recent news flow (VBTS, Close Monitoring Market,
default, regulatory review by the Capital Markets Board) — statistics alone do not
establish cause and effect.

---

## 5. The weekly volume-return relationship is regime-dependent — don't trust a single "average" result

**Finding:** With `analyze_weekly_volume.py::walk_forward_regime_check`, the
`genel_fark` ("overall difference" — the return after a low-volume week minus the
return after a high-volume week) computed across 5 walk-forward folds is consistently
NEGATIVE in the first 4 folds (-0.0037, -0.0026, -0.0026, -0.0054), but turns POSITIVE
in the LAST fold (2022-06-10 → 2026-09-25) (+0.0059) —
`quanttro.diagnostics.regime.detect_regime_break` automatically detects this sign
change (t≈14.2, a clear sign reversal).

**Problem:** Applying a single Spearman correlation/t-test to the entire dataset and
reducing it to a single number — "the return after a high-volume week is X" — completely
hides this regime change. A strategy built on such an "average" result would work in
the regime seen during the exploratory period, but could run in the opposite direction
in the most recent (and most relevant) period.

**Solution:** Always examine walk-forward/fold-level results (not just a single pooled
statistic); use `quanttro.diagnostics.regime.detect_regime_break` and
`rolling_correlation_stability` to check whether the most recent period is consistent
with the previous periods or has flipped sign.

**Rule of thumb:** Before reporting an "average IC/correlation is positive" result, ask
"is this positivity consistent across all sub-periods, or was it driven by one or two
old periods?"

---

## 6. There are data errors originating from yfinance — check with the `validate_*` scripts

**Finding:** For THYAO, the bar dated **2005-01-03** in `data/daily/THYAO.csv` produces
an impossible daily return of **~981x (i.e. ~98,100%)** (Close jumping from ~0.00092 to
~0.90). This coincides exactly with the date of Turkey's New Turkish Lira (YTL)
redenomination on January 1, 2005 (dropping 6 zeros) — meaning that yfinance's
`auto_adjust`/split-adjustment mechanism did not handle this event correctly, leaving a
spurious "mega-return" bar between pre-2005 and post-2005 prices.

**Problem:** This single bar:
- Renders the entire series' non-winsorized statistics (mean return, std, Sharpe-like
  measures) meaningless (see the absurd result in the `examples/quickstart.py` output,
  "THYAO period return: 17,645,526%" — this is not a statistical "success," it is a
  sign of a data error),
- Can get miscounted as a false "limit day" by functions like
  `detect_limit_streaks`/ACF (since ret >= 0.095, it gets flagged as a limit-up day,
  even though it is not real).

**Solution:** Before using raw data, scan for extreme values with validation scripts
in the style of `validate_volume_dow_rule.py` (or a similar "is this return realistic"
check); filter or correct known redenomination/split-error bars such as THYAO
2005-01-03 **before feature computation** (a simple solution: flag bars where
`abs(ret) > some threshold (e.g. 5.0)` as data errors and convert them to NaN, or
manually review the date in question).

**Rule of thumb:** When adding a new symbol/timeframe, at minimum skim through a list
of the "largest N-day returns" (see the simple `ret.sort_values(ascending=False).head()`
check used in preparing this document) — unrealistic values usually stand out
immediately.

---

## Summary Table

| # | Topic | File/Function | Risk |
|---|------|------------------|------|
| 1 | Market-wide shock days | `feature_engineering.py` (`RETURN_CLIP`) | Statistics/model distorted by extreme outliers |
| 2 | USD/TRY trend-spurious correlation | `load_macro_aligned`, `add_relative_target` | Spurious/common-trend correlation is learned |
| 3 | Turning zero-volume into NaN | `build_features` (volume_zscore) | Rolling window distorted, information loss |
| 4 | Low-float autocorrelation | `quanttro.diagnostics.anomalies` | "Predictability" confused with manipulation/distress |
| 5 | Regime-dependent volume-return relationship | `quanttro.diagnostics.regime` | A single "average" result hides a regime break |
| 6 | yfinance data errors (THYAO 2005) | data loading / validate scripts | Entire series' statistics become meaningless |
