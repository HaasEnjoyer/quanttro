"""Model fabrikasi: regresyon/siniflandirma gorevleri icin farkli model
turlerini tek bir arayuzden uretir.

train_model.py icindeki ``make_model`` fonksiyonunun genellestirilmis hali:
orada sadece LightGBM vardi. Burada ayni muhafazakar (asiri ogrenmeyi
sinirlayan) hiperparametre felsefesi -kucuk/gurultulu finansal veri
setlerinde fazla derin/karmasik modellerden kacinma- diger model turlerine
de tasinmistir:

- "lightgbm": LightGBM (train_model.py ile ayni varsayilanlar)
- "xgboost": XGBoost (YENİ)
- "random_forest": scikit-learn RandomForest (YENİ)
- "linear": Ridge (regresyon) / LogisticRegression (siniflandirma) (YENİ)
"""
from __future__ import annotations

from typing import Any

VALID_TASKS = ("regression", "classification")


def _check_task(task: str) -> str:
    task = task.lower()
    if task not in VALID_TASKS:
        raise ValueError(f"task {task!r} gecersiz; secenekler: {VALID_TASKS}")
    return task


def _build_lightgbm(task: str, params: dict[str, Any]):
    import lightgbm as lgb

    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        verbose=-1,
    )
    defaults.update(params)
    if task == "regression":
        return lgb.LGBMRegressor(**defaults)
    return lgb.LGBMClassifier(**defaults)


def _build_xgboost(task: str, params: dict[str, Any]):
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - ortama bagli
        raise ImportError(
            "xgboost paketi kurulu degil. `pip install xgboost` ile yukleyin."
        ) from exc

    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=0,
    )
    defaults.update(params)
    if task == "regression":
        return xgb.XGBRegressor(**defaults)
    defaults.setdefault("eval_metric", "logloss")
    return xgb.XGBClassifier(**defaults)


def _build_random_forest(task: str, params: dict[str, Any]):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    defaults = dict(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=42,
    )
    defaults.update(params)
    if task == "regression":
        return RandomForestRegressor(**defaults)
    return RandomForestClassifier(**defaults)


def _build_linear(task: str, params: dict[str, Any]):
    if task == "regression":
        from sklearn.linear_model import Ridge

        defaults = dict(alpha=1.0)
        defaults.update(params)
        return Ridge(**defaults)

    from sklearn.linear_model import LogisticRegression

    defaults = dict(C=1.0, max_iter=1000)
    defaults.update(params)
    return LogisticRegression(**defaults)


_BUILDERS = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "random_forest": _build_random_forest,
    "linear": _build_linear,
}


class ModelFactory:
    """Regresyon/siniflandirma modeli ureten fabrika sinifi.

    Desteklenen ``model_type`` degerleri: "lightgbm", "xgboost",
    "random_forest", "linear" (Ridge/LogisticRegression).

    Varsayilan hiperparametreler, train_model.py'daki LightGBM
    ayarlariyla (max_depth=4, num_leaves=15, min_child_samples=50 vb.)
    aynı muhafazakar felsefeyi izler: agac derinligi/yaprak sayisi
    kisitlanir, yaprak/dugum basina minimum ornek sayisi yuksek tutulur,
    boylece az sayida hisse/gozlemle calisirken asiri ogrenme riski
    sinirlanir.
    """

    @staticmethod
    def available_models() -> tuple[str, ...]:
        return tuple(sorted(_BUILDERS))

    @classmethod
    def create(cls, model_type: str, task: str = "regression", **overrides: Any):
        """Belirtilen ``model_type`` ve ``task`` ("regression" /
        "classification") icin, varsayilan hiperparametreler uzerine
        ``overrides`` uygulanmis bir model nesnesi (fit/predict arayuzlu)
        dondurur."""
        task = _check_task(task)
        builder = _BUILDERS.get(model_type.lower())
        if builder is None:
            raise ValueError(
                f"Desteklenmeyen model_type: {model_type!r}; secenekler: {cls.available_models()}"
            )
        return builder(task, dict(overrides))


def make_model(task: str, model_type: str = "lightgbm", **overrides: Any):
    """train_model.py'daki ``make_model`` ile geriye-uyumlu kisa yol:
    varsayilan olarak LightGBM uretir, ama ``model_type`` ile baska bir
    model turu de secilebilir."""
    return ModelFactory.create(model_type=model_type, task=task, **overrides)
