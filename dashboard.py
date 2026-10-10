from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from symbols import BIST30

PRED_DIR = Path(__file__).parent / "data" / "predictions"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
TICKERS = [s.replace(".IS", "") for s in BIST30]

st.set_page_config(page_title="BIST30 Tahmin Dashboard", layout="wide")


DOW_NAMES = {0: "Pazartesi", 1: "Sali", 2: "Carsamba", 3: "Persembe", 4: "Cuma"}
SIGNAL_DOWS = {0, 1}  # walk-forward'da dogrulanan gunler


@st.cache_data
def load_ic(timeframe: str, horizon: str) -> pd.DataFrame:
    path = PRED_DIR / f"{timeframe}_ranking_{horizon}_ic.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def load_rule_validation() -> pd.DataFrame:
    path = PRED_DIR / "volume_dow_rule_validation.csv"
    return pd.read_csv(path)


@st.cache_data
def compute_todays_signals() -> pd.DataFrame:
    """Her hisse icin en son bar: haftanin gunu Pazartesi/Sali mi ve hacim kendi
    tarihsel dagilimina gore ust tertile'da mi? Bu kural 5 walk-forward fold'un
    hepsinde ayni yonde (pozitif) cikmisti - bkz. validate_volume_dow_rule.py"""
    rows = []
    for ticker in TICKERS:
        path = FEATURES_DIR / "daily" / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        low_q, high_q = df["volume_zscore"].quantile([1 / 3, 2 / 3])
        last = df.iloc[-1]
        dow = int(last["day_of_week"])
        is_high_vol = last["volume_zscore"] >= high_q
        rows.append({
            "symbol": ticker, "tarih": df.index[-1].date(), "gun": DOW_NAMES.get(dow, dow),
            "hacim_zscore": last["volume_zscore"], "yuksek_hacim": is_high_vol,
            "sinyal": bool(dow in SIGNAL_DOWS and is_high_vol),
        })
    return pd.DataFrame(rows)


@st.cache_data
def load_ranking_predictions(timeframe: str, horizon: str) -> pd.DataFrame:
    path = PRED_DIR / f"{timeframe}_ranking_{horizon}_predictions.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def load_importance(timeframe: str, horizon: str) -> pd.DataFrame:
    path = PRED_DIR / f"{timeframe}_ranking_{horizon}_importance.csv"
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_fold_metrics(timeframe: str, label: str) -> pd.DataFrame:
    path = PRED_DIR / f"{timeframe}_{label}_fold_metrics.csv"
    return pd.read_csv(path)


@st.cache_data
def load_price(timeframe: str, ticker: str) -> pd.DataFrame:
    path = FEATURES_DIR / timeframe / f"{ticker}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def summary_row(timeframe: str, horizon: str) -> dict:
    ic_df = load_ic(timeframe, horizon)
    return {
        "Zaman dilimi": timeframe, "Ufuk": horizon,
        "Ortalama IC": ic_df["ic"].mean(),
        "IC_IR": ic_df["ic"].mean() / ic_df["ic"].std(),
        "Pozitif IC orani": (ic_df["ic"] > 0).mean(),
        "Periyot sayisi": len(ic_df),
    }


st.title("BIST30 Sinyal & Tahmin Dashboard")
st.caption(
    "Fiyat/hacim/makro ozelliklerinden uretilen kesitsel siralama sinyalinin "
    "walk-forward dogrulama sonuclari. yfinance verisi, LightGBM model."
)

tab_signals, tab_summary, tab_ranking, tab_stock, tab_models, tab_regime = st.tabs(
    ["Bugunun Sinyalleri", "Ozet", "Siralama Sinyali", "Hisse Bazli",
     "Model Karsilastirmasi", "Hacim Rejim Kirilmasi"]
)

# ---------------- TAB 0: BUGUNUN SINYALLERI ----------------
with tab_signals:
    st.subheader("Pazartesi/Sali + yuksek hacim kurali")
    rule_val = load_rule_validation()
    rule_summary = rule_val.groupby("gun").agg(
        ortalama_fark=("fark", "mean"),
        pozitif_fold_orani=("fark", lambda x: (x > 0).mean()),
    )

    st.markdown(
        "Bu kural **5 walk-forward fold'un hepsinde** (out-of-sample, gelecek "
        "sizdirmadan) ayni yonde cikti: ilgili gunde hacmi kendi tarihsel "
        "dagilimina gore en ust tertile'da olan hisseler, ertesi gun ortalamada "
        "daha iyi performans gosterdi."
    )
    c1, c2 = st.columns(2)
    for col, dow_name in zip([c1, c2], ["Pazartesi", "Sali"]):
        row = rule_summary.loc[dow_name]
        col.metric(
            f"{dow_name}: ort. fark (yuksek - dusuk hacim)",
            f"{row['ortalama_fark']*10000:.1f} bps",
            help="5 out-of-sample fold ortalamasi. Pozitif fold orani: "
                 f"{row['pozitif_fold_orani']*100:.0f}%",
        )

    st.markdown("---")
    signals = compute_todays_signals()
    data_date = signals["tarih"].max()
    flagged = signals[signals["sinyal"]].sort_values("hacim_zscore", ascending=False)

    st.markdown(f"**Son mevcut veri tarihi: {data_date}** (fetch_data.py ile guncellenebilir)")

    if len(flagged) == 0:
        st.warning(
            "Son veri tarihinde kurala uyan hisse yok (gun Pazartesi/Sali degil "
            "veya hic bir hissede hacim ust tertile'da degil)."
        )
    else:
        st.success(f"{len(flagged)} hisse kurala uyuyor:")
        st.dataframe(
            flagged[["symbol", "gun", "hacim_zscore"]].rename(columns={
                "symbol": "Hisse", "gun": "Gun", "hacim_zscore": "Hacim z-skoru",
            }),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Bu bir 'kesin al' sinyali DEGILDIR. Tarihsel ortalama etki kucuk "
        "(~12-16 baz puan) ve islem maliyeti dahil degil — olasiliksal bir "
        "egilimi gosterir, garanti degil."
    )

# ---------------- TAB 1: OZET ----------------
with tab_summary:
    st.subheader("Kesitsel siralama sinyali — ozet")
    rows = [summary_row(tf, h) for tf in ["daily", "hourly"] for h in ["h1", "h5"]]
    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.format({
            "Ortalama IC": "{:.4f}", "IC_IR": "{:.3f}", "Pozitif IC orani": "{:.1%}",
        }),
        use_container_width=True,
    )
    best = summary_df.loc[summary_df["Ortalama IC"].idxmax()]
    st.info(
        f"En guclu sinyal: **{best['Zaman dilimi']} / {best['Ufuk']}** "
        f"(IC={best['Ortalama IC']:.4f}, pozitif IC orani %{best['Pozitif IC orani']*100:.1f}). "
        "Butun degerler zayif ama tutarli pozitif — rastgeleden ayrisiyor, ama "
        "guclu bir tahmin gucu degil."
    )

    st.subheader("Yon tahmini (tek hisse, siniflandirma) — baseline karsilastirmasi")
    cols = st.columns(2)
    for col, tf in zip(cols, ["daily", "hourly"]):
        fm = load_fold_metrics(tf, "full_classification_detrended")
        with col:
            st.markdown(f"**{tf.capitalize()}**")
            st.dataframe(
                fm.style.format({"accuracy": "{:.3f}", "baseline_acc": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"Ortalama dogruluk: {fm['accuracy'].mean():.3f} vs "
                f"baseline: {fm['baseline_acc'].mean():.3f}"
            )

# ---------------- TAB 2: SIRALAMA SINYALI ----------------
with tab_ranking:
    c1, c2 = st.columns(2)
    timeframe = c1.selectbox("Zaman dilimi", ["daily", "hourly"], key="rank_tf")
    horizon = c2.selectbox(
        "Ufuk", ["h1", "h5"], key="rank_h",
        format_func=lambda h: "1 bar ileri" if h == "h1" else "5 bar ileri",
    )

    ic_df = load_ic(timeframe, horizon).sort_index()
    mean_ic = ic_df["ic"].mean()
    ic_ir = mean_ic / ic_df["ic"].std()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ortalama IC", f"{mean_ic:.4f}")
    m2.metric("IC_IR", f"{ic_ir:.3f}")
    m3.metric("Pozitif IC orani", f"%{(ic_df['ic'] > 0).mean()*100:.1f}")
    m4.metric("Periyot/ortalama spread", f"{ic_df['spread'].mean():.5f}")

    st.markdown("**Ust-alt tertile long-short — kumulatif getiri (maliyetsiz, varsayimsal)**")
    equity = (1 + ic_df["spread"]).cumprod() - 1
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Long-short"))
    fig_eq.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".1%")
    st.plotly_chart(fig_eq, use_container_width=True)
    st.caption("Islem maliyeti/kayma dahil degil — sadece sinyal yonunun tutarliligini gosterir.")

    st.markdown("**Rolling IC (60 periyot pencere)**")
    rolling_ic = ic_df["ic"].rolling(60).mean()
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic.values, mode="lines", name="Rolling IC"))
    fig_ic.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_ic.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ic, use_container_width=True)

    st.markdown("**Ozellik onemi**")
    importance = load_importance(timeframe, horizon).sort_values("importance")
    fig_imp = go.Figure(go.Bar(x=importance["importance"], y=importance.index, orientation="h"))
    fig_imp.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------------- TAB 3: HISSE BAZLI ----------------
with tab_stock:
    c1, c2, c3 = st.columns(3)
    s_timeframe = c1.selectbox("Zaman dilimi", ["daily", "hourly"], key="stock_tf")
    s_horizon = c2.selectbox(
        "Ufuk", ["h1", "h5"], key="stock_h",
        format_func=lambda h: "1 bar ileri" if h == "h1" else "5 bar ileri",
    )
    ticker = c3.selectbox("Hisse", TICKERS, key="stock_ticker")

    price = load_price(s_timeframe, ticker)
    preds = load_ranking_predictions(s_timeframe, s_horizon)
    stock_preds = preds[preds["symbol"] == ticker].sort_index()

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=price.index, y=price["close"], mode="lines", name="Kapanis"))
    fig_price.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        title=f"{ticker} fiyat (test donemi disinda da gosterilir)",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    if len(stock_preds) == 0:
        st.warning("Bu hisse icin test donemi tahmini yok (walk-forward ilk fold'da egitim verisinde kalmis olabilir).")
    else:
        hit_rate = (
            (stock_preds["predicted"] > 0) == (stock_preds["actual"] > 0)
        ).mean()
        st.metric(f"{ticker} icin yon isabet orani (test donemi)", f"%{hit_rate*100:.1f}")

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=stock_preds["predicted"], y=stock_preds["actual"], mode="markers",
            marker=dict(size=5, opacity=0.5), name=ticker,
        ))
        fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_scatter.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tahmin edilen bagil getiri", yaxis_title="Gerceklesen bagil getiri",
            title="Kalibrasyon: tahmin vs. gerceklesen (test donemi)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- TAB 4: MODEL KARSILASTIRMASI ----------------
with tab_models:
    st.subheader("Regresyon (tek hisse getiri buyuklugu) — detrended hedef")
    cols = st.columns(2)
    for col, tf in zip(cols, ["daily", "hourly"]):
        fm = load_fold_metrics(tf, "core_regression_detrended")
        with col:
            st.markdown(f"**{tf.capitalize()}**")
            st.dataframe(
                fm.style.format({"mae": "{:.5f}", "baseline_mae": "{:.5f}", "directional_acc": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                f"Yon dogrulugu ort.: {fm['directional_acc'].mean():.3f} "
                f"(rastgele=0.500) — MAE model/baseline: "
                f"{fm['mae'].mean():.5f} / {fm['baseline_mae'].mean():.5f}"
            )

    st.markdown("---")
    st.markdown(
        "**Okuma notu:** Tek-hisse regresyon/siniflandirma sonuclari rastgeleden "
        "ayrisamiyor (bkz. yon dogrulugu ~0.50). Kesitsel siralama (Ozet ve "
        "Siralama Sinyali sekmeleri) ise zayif ama tutarli pozitif bir sinyal "
        "gosteriyor. Bu, modelin 'bu hisse yarin yukselir mi' sorusuna degil, "
        "'bu hisseler arasinda hangisi digerinden iyi gider' sorusuna zayif da "
        "olsa cevap verebildigini gosteriyor."
    )

# ---------------- TAB 5: HACIM REJIM KIRILMASI ----------------
with tab_regime:
    st.subheader("Haftalik hacim kurallari — zaman icinde tutarlilik")
    st.caption(
        "Iki ayri haftalik hacim kurali (momentum: yukari+yuksek hacim vs "
        "yukari+dusuk hacim; genel: dusuk vs yuksek hacim) 5 out-of-sample "
        "fold'da test edildi. Her iki kural da son fold'da (2022 sonrasi) "
        "ters yone donuyor — ayni zamanda, bagimsiz iki kuralda."
    )

    regime = pd.read_csv(
        Path(__file__).parent / "data" / "predictions" / "weekly_volume_regime.csv"
    )
    regime["donem"] = regime["fold_start"].astype(str) + " / " + regime["fold_end"].astype(str)

    fig_regime = go.Figure()
    fig_regime.add_trace(go.Bar(
        x=regime["donem"], y=regime["momentum_fark"], name="Momentum kurali (yukari: yuksek-dusuk hacim)",
        marker_color=["#2a9d8f" if v > 0 else "#e76f51" for v in regime["momentum_fark"]],
    ))
    fig_regime.add_hline(y=0, line_color="gray", line_dash="dot")
    fig_regime.update_layout(
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        title="Momentum kurali: fold basina etki (pozitif = kural tutuyor)",
        yaxis_tickformat=".2%",
    )
    st.plotly_chart(fig_regime, use_container_width=True)

    fig_regime2 = go.Figure()
    fig_regime2.add_trace(go.Bar(
        x=regime["donem"], y=regime["genel_fark"], name="Genel kural (dusuk-yuksek hacim)",
        marker_color=["#2a9d8f" if v > 0 else "#e76f51" for v in regime["genel_fark"]],
    ))
    fig_regime2.add_hline(y=0, line_color="gray", line_dash="dot")
    fig_regime2.update_layout(
        height=360, margin=dict(l=10, r=10, t=40, b=10),
        title="Genel kural: fold basina etki (pozitif = dusuk hacim daha iyi)",
        yaxis_tickformat=".2%",
    )
    st.plotly_chart(fig_regime2, use_container_width=True)

    st.warning(
        "Her iki grafikte de son fold (2022 sonrasi) onceki 4 fold'a gore ters "
        "renkte/yonde — bu rastgele gurultu degil, BIST'teki hacim-getiri "
        "iliskisinin yuksek enflasyon/devaluasyon rejiminde yapisal olarak "
        "degistigine isaret ediyor. Bu kurallari canli kullanmadan once rejim "
        "degiskeni (orn. yillik enflasyon, USD/TRY volatilitesi) ile filtrelemek "
        "gerekir."
    )
