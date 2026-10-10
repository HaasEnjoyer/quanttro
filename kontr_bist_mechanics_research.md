# BIST Tavan/Taban Mechanics and the KONTR Case — Research Report

*Research conducted via Turkish and English web sources, October 2026. Confidence levels are noted per section; where sourcing was thin, that is stated explicitly rather than inferred.*

## 1. BIST Price Limit Mechanics

**Core rule:** Borsa İstanbul applies a daily price band ("fiyat marjı") of ±10% around the previous close for ordinary equity markets (Yıldız Pazar / Star Market and Ana Pazar / Main Market). This is the "tavan" (ceiling, +10%) and "taban" (floor, −10%). No confirmed 2024–2026 blanket change to the standard ±10% band was found.

**Compounding, not simple addition:** A limit-up streak compounds off the prior day's new price. An "8-day tavan serisi" implies roughly 1.10^8 ≈ 2.14x cumulative appreciation, not 80%.

**What happens when a stock is stuck at the limit:** Once price hits the ceiling, the stock continues in continuous auction *at that exact price only* — buy orders queue in time priority; any seller willing to sell at that fixed price gets matched. If no sellers appear, orders sit unfilled all day.

**Market-tier differences:** Segmentation (Yıldız Pazar, Ana Pazar, GİP, Yakın İzleme Pazarı/YİP) is mainly by free-float market cap and listing standards. The daily ±10% band applies uniformly across Yıldız/Ana/YİP, though YİP replaces continuous trading with a single-price (tek fiyat) auction five times a day and disallows margin trading and short selling.

**Confidence:** Medium-high on the ±10% figure; lower on tier-specific nuances.

## 2. Documented Tavan/Taban Streak Cases on BIST

Example found: **1000 Yatırımlar Holding (BINHO)** hit the ceiling 7 consecutive sessions after listing (~94% cumulative rise), with ~8 million lots queued unfilled at the ceiling on the final day.

**Much bigger context — the 2025–2026 "fon krizi" (fund scandal):** Per Bloomberg (relayed by Euronews Türkçe, ParaAnaliz, Medyascope), roughly **60 companies since 2023** — generally recent IPOs with low public float concentrated in a handful of investment funds — saw prices driven to levels unexplainable by fundamentals before collapsing. Combined investor losses estimated near **$16 billion**; ~130 funds subject to liquidation/wind-down. Suspected mechanism: coordinated funds trading among themselves (and affiliated companies) to inflate thinly-floated names. **KONTR's behavior should likely be read against this backdrop — one case among a broader, now-regulatorily-acknowledged class of low-float manipulation patterns.**

## 3. KONTR-Specific Regulatory History

- **SPK fine over IPO process** (pre-2024): SPK fined Kontrolmatik, intermediary Gedik Yatırım, board chair Sami Aslanhan, and board member Ömer Ünsalan ~563,496 TL combined.
- **VBTS measures, April 21, 2026**: After KONTR hit the floor (-10% to 12.99 TL after a month of ~40% losses), SPK/BIST imposed Level-1 VBTS (short-sale/margin ban), effective April 22–May 21, 2026.
- **Moved to Yakın İzleme Pazarı, June 17–18, 2026**: Trading halted then resumed in the distressed-stock market. Reason: KONTR **defaulted twice** on bond interest/principal payments (~450M TL) in May and June 2026. **Removed from BIST 50, BIST 100, and BIST 500 indices** as a direct result.
- **Unresolved allegations** (ParaMedya investigative piece, not SPK-adjudicated): questions over $300-400M raised via IPO/secondary sales/bonds, accusations of long-standing "piyasa oyunculuğu" (market-gaming) against chairman Sami Aslanhan, claims IPO proceeds partly used for share-price support coordinated with market operators, related-party exposure. CEO publicly said "I'm the one most upset" about the decline, then sold 79 million lots ~2 weeks later.
- **Separate Ministry of Labor investigation** over delayed employee salary payments.

**Confidence:** High on VBTS/YİP/index-removal timeline; medium on fraud allegations (single-source journalism, not independently verified).

## 4. KONTR Free Float / Ownership Concentration — key correction

**KONTR's free float is currently ~88%** (high, not low) as of late 2025/2026. At IPO (2020) float was only 20% — genuinely low, fitting a "low-float squeeze" profile early on. Chairman Sami Aslanhan's stake fell from ~40% at listing → 29.09% (end-2022) → **10.02% (Dec 22, 2025)** after a wholesale sale of shares explicitly reported as having "increased public float and created sustained selling pressure."

**Implication:** The tavan-streak statistic likely reflects KONTR's **early-listing period (2020–2022, float 20-29%)**, not its current float profile. The 2025-2026 period shows the *opposite* pattern — taban (floor) streaks tied to distress/default, not a pump. The autocorrelation finding should be date-partitioned to separate these two very different regimes.

## 5. Fintwit / Retail Trader Lore on "Oyuncu" Behavior

Low-medium confidence, scattered sourcing: **"oyuncu var" / "sahipli mal"** ("owned goods") describes a stock believed actively driven by a concentrated operator; **"kademe"** (price-level) watching — traders watch order-book depth for signs of a defended level; **"çene"** — opportunists who ride an operator's pattern. Standard warning across sources: these patterns break without notice, leaving latecomers "trapped" with no bid — which maps directly onto what happened to KONTR in April–June 2026.

## Summary of Key Caveats

- The "7.7% of days / 8-day streak" KONTR statistic is our own backtest finding — no external source corroborates or contradicts that specific number.
- KONTR's free float is currently high (~88%); the low-float period was mainly 2020–2022, with a renewed selling-pressure/high-float phase beginning late 2025.
- The 2025–2026 "fon krizi" (~60 companies, ~$16B losses) is the key context for low-float manipulation patterns on BIST in this era — KONTR's own 2026 VBTS measures, index removals, and bond default sit squarely inside that period.
- Manipulation allegations against Kontrolmatik's management are open questions from investigative journalism, not SPK-confirmed findings.
