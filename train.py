import runtime_env  # noqa: F401 — OpenMP shim before numpy/sklearn/torch

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import CFG
from features import build_preprocessor
from models.ensemble import build_ensemble_from_config
from pipeline.feature_engineering import engineer_features

try:
    from models.tabular_transformer import fit_tabular_transformer

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    fit_tabular_transformer = None  # type: ignore[assignment, misc]

try:
    from anomaly_autoencoder import fit_autoencoder_anomaly

    _AE_AVAILABLE = True
except ImportError:
    _AE_AVAILABLE = False
    fit_autoencoder_anomaly = None  # type: ignore[assignment, misc]

ANOMALY_RESULT_KEYS = frozenset({"isoforest", "autoencoder"})


def _joblib_dump(obj: object, path: Path) -> None:
    lvl = int(getattr(CFG, "JOBLIB_COMPRESS", 0) or 0)
    if lvl > 0:
        joblib.dump(obj, path, compress=lvl)
    else:
        joblib.dump(obj, path)


def _binary_neg_pos_counts(y: pd.Series) -> tuple[int, int]:
    """Return (n_class_0, n_class_1) for binary integer labels; missing labels count as 0."""
    vc = y.astype(int).value_counts()
    return int(vc.get(0, 0)), int(vc.get(1, 0))


def _xgb_scale_pos_weight(y_train: pd.Series) -> float:
    n0, n1 = _binary_neg_pos_counts(y_train)
    return float(n0 / max(1, n1))


def _roc_auc_safe(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _swap_xy_rows(
    X_left: pd.DataFrame,
    X_right: pd.DataFrame,
    y_left: pd.Series,
    y_right: pd.Series,
    i_left: int,
    i_right: int,
) -> None:
    """Swap one row between left and right frames (in place)."""
    tmp_x = X_left.iloc[i_left].copy()
    X_left.iloc[i_left] = X_right.iloc[i_right]
    X_right.iloc[i_right] = tmp_x
    tmp_y = int(y_left.iloc[i_left])
    y_left.iloc[i_left] = int(y_right.iloc[i_right])
    y_right.iloc[i_right] = tmp_y


def _build_lr_calibrator(lr_pipe: Pipeline, y_train: pd.Series) -> Pipeline | CalibratedClassifierCV:
    """
    Isotonic calibration uses internal stratified CV; each class must have at least
    as many samples as ``cv`` folds (sklearn raises if not). On tiny training sets,
    fall back to the raw logistic pipeline.
    """
    n0, n1 = _binary_neg_pos_counts(y_train)
    min_c = min(n0, n1)
    if min_c < 2:
        warnings.warn(
            "Using logistic regression without isotonic calibration: the smaller "
            f"training class has only {min_c} sample(s) (legit={n0}, fraud={n1}).",
            UserWarning,
            stacklevel=2,
        )
        return lr_pipe
    n_splits = int(min(3, min_c))
    if n_splits < 2:
        return lr_pipe
    return CalibratedClassifierCV(lr_pipe, method="isotonic", cv=n_splits)


def _rebalance_train_test_for_binary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    pos: int = 1,
    neg: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified splits with very few positives can put **all** fraud rows in train
    and none in test (or the reverse). Swap one positive with one negative across
    the boundary so both sets contain both classes when possible.
    """
    X_tr = X_train.reset_index(drop=True).copy()
    X_te = X_test.reset_index(drop=True).copy()
    y_tr = y_train.reset_index(drop=True).astype(int).copy()
    y_te = y_test.reset_index(drop=True).astype(int).copy()

    if y_tr.nunique() >= 2 and y_te.nunique() >= 2:
        return X_tr, X_te, y_tr, y_te

    fixed = False
    # Test has no fraud: move one fraud from train, swap with a test legitimate row
    if (y_te == pos).sum() == 0 and (y_tr == pos).sum() >= 1:
        i_tr = int(np.flatnonzero((y_tr == pos).to_numpy())[0])
        i_te = int(np.flatnonzero((y_te == neg).to_numpy())[0])
        _swap_xy_rows(X_tr, X_te, y_tr, y_te, i_tr, i_te)
        fixed = True
    # Train has no fraud: move one fraud from test
    elif (y_tr == pos).sum() == 0 and (y_te == pos).sum() >= 1:
        i_te = int(np.flatnonzero((y_te == pos).to_numpy())[0])
        i_tr = int(np.flatnonzero((y_tr == neg).to_numpy())[0])
        _swap_xy_rows(X_tr, X_te, y_tr, y_te, i_tr, i_te)
        fixed = True

    if fixed:
        warnings.warn(
            "Adjusted train/test rows so both splits contain fraud and legitimate "
            "labels (tiny positive counts can otherwise leave one split single-class).",
            UserWarning,
            stacklevel=2,
        )
    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        raise ValueError(
            f"Could not form train/test splits with both classes in each set. "
            f"Train: legit={_binary_neg_pos_counts(y_tr)[0]}, fraud={_binary_neg_pos_counts(y_tr)[1]}; "
            f"Test: legit={_binary_neg_pos_counts(y_te)[0]}, fraud={_binary_neg_pos_counts(y_te)[1]}. "
            "Add more minority-class rows (e.g. larger --n in sample-transactions)."
        )
    return X_tr, X_te, y_tr, y_te


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def infer_columns(df: pd.DataFrame):
    categorical_cols = [c for c in CFG.CATEGORICAL_FEATURES if c in df.columns]

    v_cols = [c for c in df.columns if c.startswith("V")]
    numeric_cols = [
        "amount",
        "time_since_prev_tx",
        "rolling_tx_count_1h",
        "rolling_amount_sum_1h",
        "rolling_tx_count_24h",
        "rolling_amount_sum_24h",
        "amount_zscore_24h",
        "hour",
        "day_of_week",
        "is_weekend",
        "rolling_amount_mean_5tx",
        "rolling_amount_mean_10tx",
        "amount_deviation_user_mean",
    ] + v_cols

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    return numeric_cols, categorical_cols


def train_models(df: pd.DataFrame, *, use_velocity_features: bool = True):
    df = df.dropna(subset=[CFG.TARGET_COL])
    df = engineer_features(
        df,
        CFG.TIME_COL,
        CFG.USER_COL,
        CFG.AMOUNT_COL,
        use_velocity=use_velocity_features,
    )
    y = df[CFG.TARGET_COL].astype(int)
    X = df.drop(columns=[CFG.TARGET_COL])

    n0, n1 = _binary_neg_pos_counts(y)
    if n1 == 0 or n0 == 0:
        raise ValueError(
            f"{CFG.TARGET_COL} must contain both classes 0 and 1. "
            f"Found legitimate={n0}, fraud={n1}. "
            "Resample from the full CSV, e.g. "
            "`python scripts.py sample-transactions --n 100 --stratify-col is_fraud`."
        )

    numeric_cols, categorical_cols = infer_columns(df)
    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # sklearn needs >=2 samples per class to stratify; otherwise split is random
    stratify_arg = y if min(n0, n1) >= 2 else None
    if stratify_arg is None:
        warnings.warn(
            f"Stratified split disabled: rarest class count is {min(n0, n1)} (<2). "
            "Train/test may be imbalanced; add more minority-class rows if metrics look wrong.",
            UserWarning,
            stacklevel=2,
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.TEST_SIZE, random_state=CFG.RANDOM_STATE, stratify=stratify_arg
    )
    X_train, X_test, y_train, y_test = _rebalance_train_test_for_binary(
        X_train, X_test, y_train, y_test
    )

    # Class weights (cost-sensitive baseline)
    classes = np.array(sorted(y_train.unique()))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {cls: w for cls, w in zip(classes, cw)}

    fast = bool(getattr(CFG, "FAST_TRAIN", False))
    rf_trees = 80 if fast else 300
    iso_trees = 100 if fast else 300
    gbm_trees = 80 if fast else 200
    xgb_trees = 150 if fast else 400
    lgb_trees = 150 if fast else 400
    lr_max_iter = 2000 if fast else 8000
    ae_epochs = int(getattr(CFG, "FAST_AUTOENCODER_EPOCHS", 8)) if fast else int(
        getattr(CFG, "AUTOENCODER_EPOCHS", 25)
    )
    if fast:
        print("FAST_TRAIN=True: using smaller forests/boosting rounds and fewer AE epochs.")

    # 1) Logistic Regression baseline (calibrated probabilities)
    lr = LogisticRegression(max_iter=lr_max_iter, class_weight=class_weight, solver="lbfgs")
    lr_pipe = Pipeline([("pre", preprocessor), ("clf", lr)])
    lr_cal = _build_lr_calibrator(lr_pipe, y_train)

    # 2) Decision Tree (baseline, interpretable)
    dt = DecisionTreeClassifier(
        max_depth=16,
        min_samples_leaf=20,
        random_state=CFG.RANDOM_STATE,
        class_weight=class_weight,
    )
    dt_pipe = Pipeline([("pre", preprocessor), ("clf", dt)])

    # 3) Random Forest
    rf = RandomForestClassifier(
        n_estimators=rf_trees, random_state=CFG.RANDOM_STATE, class_weight=class_weight, n_jobs=-1
    )
    rf_pipe = Pipeline([("pre", preprocessor), ("clf", rf)])

    # 4) Gradient Boosting (sklearn GBM — thesis methodology)
    gbm = GradientBoostingClassifier(
        n_estimators=gbm_trees,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=CFG.RANDOM_STATE,
    )
    gbm_pipe = Pipeline([("pre", preprocessor), ("clf", gbm)])

    # 5) XGBoost
    xgb = XGBClassifier(
        n_estimators=xgb_trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=CFG.RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=_xgb_scale_pos_weight(y_train),
    )
    xgb_pipe = Pipeline([("pre", preprocessor), ("clf", xgb)])

    # 6) LightGBM (gradient boosting — thesis methodology)
    lgbm = LGBMClassifier(
        n_estimators=lgb_trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        class_weight=class_weight,
    )
    lgbm_pipe = Pipeline([("pre", preprocessor), ("clf", lgbm)])

    # 7) SMOTE + Logistic Regression (imbalance-aware)
    smote_lr = ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=CFG.RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=lr_max_iter, solver="lbfgs"))
    ])

    rus_lr = None
    if CFG.UNDERSAMPLE_ENABLED:
        rus_lr = ImbPipeline([
            ("pre", preprocessor),
            ("rus", RandomUnderSampler(random_state=CFG.RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=lr_max_iter, solver="lbfgs")),
        ])

    # Unsupervised anomaly detector (IsolationForest) — trains on non-fraud only
    iso = IsolationForest(n_estimators=iso_trees, random_state=CFG.RANDOM_STATE, contamination="auto")
    # anomaly uses numeric-only in a simple way; for real work, do better encoding
    # Here we just use numeric cols after feature engineering
    Xn_train = X_train[numeric_cols].fillna(0.0)
    Xn_test = X_test[numeric_cols].fillna(0.0)

    sample_weight_balanced = compute_sample_weight(class_weight="balanced", y=y_train)

    # Fit supervised models
    lr_cal.fit(X_train, y_train)
    dt_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    gbm_pipe.fit(X_train, y_train, clf__sample_weight=sample_weight_balanced)
    xgb_pipe.fit(X_train, y_train)
    lgbm_pipe.fit(X_train, y_train)
    smote_fitted = False
    try:
        smote_lr.fit(X_train, y_train)
        smote_fitted = True
    except ValueError as exc:
        print(f"SMOTE + LR skipped (needs >=2 classes in train): {exc}")
    if rus_lr is not None:
        rus_lr.fit(X_train, y_train)

    ensemble_model = None
    try:
        ensemble_model = build_ensemble_from_config(lr_cal, rf_pipe, xgb_pipe)
        if ensemble_model is not None:
            ensemble_model.fit(X_train, y_train)
            print("Ensemble (LR+RF+XGB) training finished.")
    except Exception as exc:
        print(f"Ensemble skipped: {exc}")
        ensemble_model = None

    tab_transformer = None
    if CFG.TRANSFORMER_ENABLED and _TORCH_AVAILABLE and fit_tabular_transformer is not None:
        try:
            tab_transformer = fit_tabular_transformer(
                X_train, y_train, preprocessor, cfg=CFG
            )
            print("Tabular Transformer training finished.")
        except Exception as exc:
            print(f"Tabular Transformer skipped: {exc}")

    # Fit anomaly model on genuine transactions (y=0)
    iso.fit(Xn_train[y_train == 0])

    autoencoder_scorer = None
    if CFG.AUTOENCODER_ENABLED and _AE_AVAILABLE and fit_autoencoder_anomaly is not None:
        try:
            autoencoder_scorer = fit_autoencoder_anomaly(
                Xn_train[y_train == 0],
                numeric_cols,
                epochs=ae_epochs,
                batch_size=min(4096, max(256, len(Xn_train))),
            )
            print("Autoencoder anomaly model training finished.")
        except Exception as exc:
            print(f"Autoencoder anomaly skipped: {exc}")

    # Evaluate
    results: dict[str, dict[str, float]] = {}
    eval_data: dict[str, object] = {
        "y_test": y_test.to_numpy(),
        "supervised": {},
        "iso_scores": None,
    }

    supervised_estimators = {
        "lr_cal": lr_cal,
        "dt": dt_pipe,
        "rf": rf_pipe,
        "gbm": gbm_pipe,
        "xgb": xgb_pipe,
        "lgbm": lgbm_pipe,
    }
    if smote_fitted:
        supervised_estimators["smote_lr"] = smote_lr
    if rus_lr is not None:
        supervised_estimators["rus_lr"] = rus_lr
    if ensemble_model is not None:
        supervised_estimators["ensemble"] = ensemble_model
    if tab_transformer is not None:
        supervised_estimators["tab_transformer"] = tab_transformer

    for name, model in supervised_estimators.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= CFG.ALERT_THRESHOLD).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        fpr = 0.0
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn + 1e-9))
        yt = y_test.to_numpy()
        results[name] = {
            "Accuracy": float(accuracy_score(y_test, preds)),
            "PR_AUC": float(average_precision_score(yt, probs))
            if len(np.unique(yt)) > 1
            else float("nan"),
            "ROC_AUC": _roc_auc_safe(yt, probs),
            "F1": float(f1_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, zero_division=0)),
            "Recall": float(recall_score(y_test, preds, zero_division=0)),
            "FPR": fpr,
        }
        eval_data["supervised"][name] = probs

    # Anomaly scoring: convert isolation forest score to "fraud probability-like"
    iso_scores = -iso.score_samples(Xn_test)  # higher => more anomalous
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_preds = (iso_scores >= CFG.ALERT_THRESHOLD).astype(int)
    yt = y_test.to_numpy()
    results["isoforest"] = {
        "Accuracy": float(accuracy_score(y_test, iso_preds)),
        "PR_AUC": float(average_precision_score(yt, iso_scores))
        if len(np.unique(yt)) > 1
        else float("nan"),
        "ROC_AUC": _roc_auc_safe(yt, iso_scores),
        "F1": float(f1_score(y_test, iso_preds)),
    }
    eval_data["iso_scores"] = iso_scores

    if autoencoder_scorer is not None:
        ae_raw = autoencoder_scorer.score_samples(X_test)
        ae_scores = (ae_raw - ae_raw.min()) / (ae_raw.max() - ae_raw.min() + 1e-9)
        ae_preds = (ae_scores >= CFG.ALERT_THRESHOLD).astype(int)
        results["autoencoder"] = {
            "Accuracy": float(accuracy_score(y_test, ae_preds)),
            "PR_AUC": float(average_precision_score(yt, ae_scores))
            if len(np.unique(yt)) > 1
            else float("nan"),
            "ROC_AUC": _roc_auc_safe(yt, ae_scores),
            "F1": float(f1_score(y_test, ae_preds)),
        }
        eval_data["ae_scores"] = ae_scores

    # Best supervised model by PR-AUC (anomaly-only models excluded)
    supervised_results_only = {k: v for k, v in results.items() if k not in ANOMALY_RESULT_KEYS}
    best_name = max(
        supervised_results_only,
        key=lambda k: (
            supervised_results_only[k]["PR_AUC"]
            if not np.isnan(supervised_results_only[k]["PR_AUC"])
            else -1.0
        ),
    )
    best_model = supervised_estimators.get(best_name, xgb_pipe)

    _joblib_dump(best_model, CFG.MODELS_DIR / "fraud_model.joblib")
    # Tabular export for thesis / reports (includes tab_transformer when trained)
    metrics_df = pd.DataFrame(
        [{"model": k, **v} for k, v in results.items()]
    )
    metrics_path = CFG.MODELS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    try:
        fnames = list(preprocessor.get_feature_names_out())
    except Exception:
        fnames = []

    slim = bool(getattr(CFG, "SLIM_SCORING_BUNDLE", False))
    bundle_supervised = (
        {best_name: best_model} if slim else supervised_estimators
    )
    bundle_shap = None if slim else rf_pipe
    if slim:
        print(
            "SLIM_SCORING_BUNDLE=True: bundle stores only the best supervised model "
            "(no per-model copies, no tree_pipeline_for_shap). Set SLIM_SCORING_BUNDLE=False for full bundle."
        )

    # Runtime bundle for hybrid scoring and multi-model comparison in API
    scoring_bundle = {
        "best_model_name": best_name,
        "best_model": best_model,
        "supervised_models": bundle_supervised,
        "results_metrics": results,
        "isoforest": iso,
        "autoencoder": autoencoder_scorer,
        "numeric_cols": numeric_cols,
        "anomaly_blend_iso": CFG.ANOMALY_BLEND_ISO,
        "anomaly_blend_ae": CFG.ANOMALY_BLEND_AE,
        "threshold": CFG.ALERT_THRESHOLD,
        "risk_bands": {
            "medium": CFG.MEDIUM_RISK_THRESHOLD,
            "high": CFG.HIGH_RISK_THRESHOLD,
        },
        # SHAP TreeExplainer: Random Forest pipeline (omitted when SLIM_SCORING_BUNDLE)
        "tree_pipeline_for_shap": bundle_shap,
        "shap_feature_names": fnames,
    }
    _joblib_dump(scoring_bundle, CFG.MODELS_DIR / "scoring_bundle.joblib")
    _joblib_dump(
        {"results": results, "best": best_name, "eval_data": eval_data},
        CFG.MODELS_DIR / "train_report.joblib",
    )

    print("Training results:", results)
    print("Best model (by PR-AUC among supervised):", best_name)
    print(f"\nSaved metrics table: {metrics_path}")
    print("\n=== Hold-out test metrics (threshold = {:.2f}) ===".format(CFG.ALERT_THRESHOLD))
    print(
        f"{'Model':<20} {'Accuracy':>10} {'PR-AUC':>10} {'ROC-AUC':>10} {'F1':>10}"
    )
    print("-" * 62)
    for name in sorted(results.keys()):
        m = results[name]
        print(
            f"{name:<20} {m['Accuracy']:>10.4f} {m['PR_AUC']:>10.4f} "
            f"{m['ROC_AUC']:>10.4f} {m['F1']:>10.4f}"
        )

if __name__ == "__main__":
    from dataset_loaders import load_dataset

    path = str(CFG.TRAIN_CSV)
    df = load_dataset(path, CFG.DATASET_PRESET)
    train_models(df)