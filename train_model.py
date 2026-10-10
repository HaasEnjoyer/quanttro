from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from symbols import BIST30

FEATURES_DIR = Path(__file__).parent / "data" / "features"
PRED_DIR = Path(__file__).parent / "data" / "predictions"

CORE_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "volume_zscore", "realized_vol_10", "atr_14",
    "rel_strength", "at_limit", "day_of_week",
]
MACRO_FEATURES = ["usdtry_ret_1_detrended", "usdtry_vol_10"]
TARGET_COL = "target_fwd_rel_ret"
N_FOLDS = 5


def load_pooled(timeframe: str) -> pd.DataFrame:
    frames = []
    for symbol in BIST30:
        ticker = symbol.replace(".IS", "")
        path = FEATURES_DIR / timeframe / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df["symbol"] = ticker
        frames.append(df)
    pooled = pd.concat(frames)
    pooled = pooled.sort_index()
    return pooled


def walk_forward_folds(dates: pd.DatetimeIndex, n_folds: int):
    unique_dates = dates.unique().sort_values()
    fold_edges = np.array_split(unique_dates, n_folds + 1)
    for i in range(1, n_folds + 1):
        train_cutoff = fold_edges[i - 1][-1]
        test_dates = fold_edges[i]
        yield train_cutoff, test_dates[0], test_dates[-1]


def naive_baseline_mae(y_true: pd.Series) -> float:
    return y_true.abs().mean()


def make_model(task: str):
    if task == "regression":
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, verbose=-1,
        )
    return lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, verbose=-1,
    )


def run(timeframe: str, extra_cols: list[str], feature_cols: list[str], task: str, label: str):
    pooled = load_pooled(timeframe)
    feature_cols = feature_cols + extra_cols

    if task == "classification":
        pooled = pooled[pooled[TARGET_COL] != 0]
        y_all = (pooled[TARGET_COL] > 0).astype(int)
    else:
        y_all = pooled[TARGET_COL]

    print(f"\n########## {timeframe.upper()} — {label} ##########")
    print(f"Ozellikler: {feature_cols}")
    print(f"Toplam satir: {len(pooled)}, hisse: {pooled['symbol'].nunique()}")

    results = []
    all_preds = []

    for train_cutoff, test_start, test_end in walk_forward_folds(pooled.index, N_FOLDS):
        train_mask = pooled.index <= train_cutoff
        test_mask = (pooled.index >= test_start) & (pooled.index <= test_end)
        train, test = pooled[train_mask], pooled[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        if len(train) < 500 or len(test) < 50:
            continue

        model = make_model(task)
        model.fit(train[feature_cols], y_train)

        if task == "classification":
            pred = model.predict(test[feature_cols])
            acc = (pred == y_test).mean()
            baseline_acc = max(y_test.mean(), 1 - y_test.mean())
            results.append({
                "test_start": test_start.date(), "test_end": test_end.date(),
                "n_test": len(test), "accuracy": acc, "baseline_acc": baseline_acc,
            })
            fold_preds = test[["symbol"]].copy()
            fold_preds["actual_up"] = y_test.values
            fold_preds["predicted_up"] = pred
        else:
            pred = model.predict(test[feature_cols])
            mae = np.abs(y_test - pred).mean()
            baseline_mae = naive_baseline_mae(y_test)
            dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
            results.append({
                "test_start": test_start.date(), "test_end": test_end.date(),
                "n_test": len(test), "mae": mae, "baseline_mae": baseline_mae,
                "directional_acc": dir_acc,
            })
            fold_preds = test[["symbol"]].copy()
            fold_preds[TARGET_COL] = y_test.values
            fold_preds["predicted"] = pred

        all_preds.append(fold_preds)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    if task == "classification":
        print(f"\nOrtalama dogruluk: {results_df['accuracy'].mean():.3f} vs baseline: {results_df['baseline_acc'].mean():.3f}")
    else:
        print(f"\nOrtalama yon dogrulugu: {results_df['directional_acc'].mean():.3f} (rastgele = 0.500)")
        print(f"Ortalama MAE: {results_df['mae'].mean():.5f} vs naive baseline: {results_df['baseline_mae'].mean():.5f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    full_preds = pd.concat(all_preds)
    full_preds.to_csv(PRED_DIR / f"{timeframe}_{label}_predictions.csv")
    results_df.to_csv(PRED_DIR / f"{timeframe}_{label}_fold_metrics.csv", index=False)

    final_model = make_model(task)
    final_model.fit(pooled[feature_cols], y_all)
    importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nOzellik onemi:")
    print(importance.to_string())


def run_ranking(timeframe: str, extra_cols: list[str], feature_cols: list[str], target_col: str, label: str):
    """Kesitsel siralama degerlendirmesi: her periyotta hisseleri tahmine gore
    sirala, Information Coefficient (Spearman) ve ust-alt tertile long-short
    spread'i olc. Tek-tek dogruluk yerine 'siralama gercekten bir seyler mi
    biliyor' sorusuna cevap arar - kucuk ama gercek bir edge varsa burada daha
    gorunur olur."""
    pooled = load_pooled(timeframe)
    feature_cols = feature_cols + extra_cols
    y_all = pooled[target_col]

    print(f"\n########## {timeframe.upper()} — {label} (ranking/IC) ##########")
    print(f"Hedef: {target_col}, Ozellikler: {feature_cols}")

    ic_rows = []
    all_preds = []

    for train_cutoff, test_start, test_end in walk_forward_folds(pooled.index, N_FOLDS):
        train_mask = pooled.index <= train_cutoff
        test_mask = (pooled.index >= test_start) & (pooled.index <= test_end)
        train, test = pooled[train_mask], pooled[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        if len(train) < 500 or len(test) < 50:
            continue

        model = make_model("regression")
        model.fit(train[feature_cols], y_train)
        pred = model.predict(test[feature_cols])

        fold = test[["symbol"]].copy()
        fold["actual"] = y_test.values
        fold["predicted"] = pred
        all_preds.append(fold)

        for ts, group in fold.groupby(fold.index):
            if len(group) < 10:
                continue
            ic = group["actual"].corr(group["predicted"], method="spearman")
            ranked = group.sort_values("predicted")
            n_tercile = max(len(ranked) // 3, 1)
            bottom = ranked.iloc[:n_tercile]["actual"].mean()
            top = ranked.iloc[-n_tercile:]["actual"].mean()
            ic_rows.append({"ts": ts, "ic": ic, "spread": top - bottom})

    ic_df = pd.DataFrame(ic_rows).dropna(subset=["ic"])
    mean_ic = ic_df["ic"].mean()
    ic_ir = mean_ic / ic_df["ic"].std() if ic_df["ic"].std() > 0 else float("nan")

    print(f"Periyot sayisi: {len(ic_df)}")
    print(f"Ortalama IC (Spearman): {mean_ic:.4f}  (IC_IR: {ic_ir:.3f})")
    print(f"Ortalama ust-alt tertile spread: {ic_df['spread'].mean():.5f} / periyot "
          f"(std: {ic_df['spread'].std():.5f})")
    print(f"Pozitif IC orani: {(ic_df['ic'] > 0).mean():.3f} (rastgele = 0.500)")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ic_df.set_index("ts").to_csv(PRED_DIR / f"{timeframe}_ranking_{label}_ic.csv")
    pd.concat(all_preds).to_csv(PRED_DIR / f"{timeframe}_ranking_{label}_predictions.csv")

    final_model = make_model("regression")
    final_model.fit(pooled[feature_cols], y_all)
    importance = pd.Series(final_model.feature_importances_, index=feature_cols, name="importance")
    importance.sort_values(ascending=False).to_csv(PRED_DIR / f"{timeframe}_ranking_{label}_importance.csv")


if __name__ == "__main__":
    # 1) sadelestirilmis ozellik seti (USD/TRY cikarildi), regresyon, bagil (excess) hedef
    run("daily", extra_cols=["is_turn_of_month"], feature_cols=CORE_FEATURES,
        task="regression", label="core_regression_detrended")
    run("hourly", extra_cols=["hour_of_day"], feature_cols=CORE_FEATURES,
        task="regression", label="core_regression_detrended")

    # 2) tam ozellik seti (detrended USD/TRY dahil), siniflandirma (yon tahmini), bagil hedef
    run("daily", extra_cols=["is_turn_of_month"], feature_cols=CORE_FEATURES + MACRO_FEATURES,
        task="classification", label="full_classification_detrended")
    run("hourly", extra_cols=["hour_of_day"], feature_cols=CORE_FEATURES + MACRO_FEATURES,
        task="classification", label="full_classification_detrended")

    # 3) kesitsel siralama (IC) + daha uzun ufuk (5-bar ileri bagil getiri)
    full_features = CORE_FEATURES + MACRO_FEATURES
    run_ranking("daily", extra_cols=["is_turn_of_month"], feature_cols=full_features,
                target_col="target_fwd_rel_ret", label="h1")
    run_ranking("daily", extra_cols=["is_turn_of_month"], feature_cols=full_features,
                target_col="target_fwd5_rel_ret", label="h5")
    run_ranking("hourly", extra_cols=["hour_of_day"], feature_cols=full_features,
                target_col="target_fwd_rel_ret", label="h1")
    run_ranking("hourly", extra_cols=["hour_of_day"], feature_cols=full_features,
                target_col="target_fwd5_rel_ret", label="h5")
