"""
Post-hoc explainability: SHAP (tree models) and LIME (tabular).
Set FRAUD_EXPLAIN_ROW=0..n-1 or pass sample CSV with one labelled row to explain.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from config import CFG
from features import engineer_features
from model_io import load_joblib


def _strip_calibrator(model):
    """Return underlying sklearn Pipeline when model is CalibratedClassifierCV."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return model.calibrated_classifiers_[0].estimator
    return model


def explain_shap_tree(sample_csv: str, row_index: int = 0, out_path: str | None = None) -> None:
    model = load_joblib(CFG.MODELS_DIR / "fraud_model.joblib", what="fraud_model.joblib")
    base = _strip_calibrator(model)
    df = pd.read_csv(sample_csv)
    if CFG.TARGET_COL in df.columns:
        X = df.drop(columns=[CFG.TARGET_COL])
    else:
        X = df
    X = engineer_features(
        X,
        CFG.TIME_COL,
        CFG.USER_COL,
        CFG.AMOUNT_COL,
        use_velocity=True,
    )
    row = X.iloc[[row_index]]

    if not hasattr(base, "named_steps"):
        print("SHAP tree path needs a sklearn Pipeline; got:", type(base))
        return
    pre = base.named_steps.get("pre")
    clf = base.named_steps.get("clf")
    if pre is None or clf is None:
        print("Pipeline missing 'pre' or 'clf' steps.")
        return
    X_trans = pre.transform(row)
    explainer = shap.TreeExplainer(clf)
    sv = explainer(X_trans)
    shap.plots.waterfall(sv[0], show=False)
    out = out_path or str(CFG.MODELS_DIR / "shap_waterfall.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP waterfall: {out}")


def explain_lime(sample_csv: str, row_index: int = 0, out_path: str | None = None) -> None:
    model = load_joblib(CFG.MODELS_DIR / "fraud_model.joblib", what="fraud_model.joblib")
    train_path = CFG.TRAIN_CSV
    if train_path.exists():
        train_ref = pd.read_csv(str(train_path))
        if CFG.TARGET_COL in train_ref.columns:
            train_ref = train_ref.drop(columns=[CFG.TARGET_COL])
        train_ref = engineer_features(
            train_ref,
            CFG.TIME_COL,
            CFG.USER_COL,
            CFG.AMOUNT_COL,
            use_velocity=True,
        )
        background = train_ref.fillna(0).head(min(5000, len(train_ref)))
    else:
        background = None

    df = pd.read_csv(sample_csv)
    if CFG.TARGET_COL in df.columns:
        X = df.drop(columns=[CFG.TARGET_COL])
    else:
        X = df
    X = engineer_features(
        X,
        CFG.TIME_COL,
        CFG.USER_COL,
        CFG.AMOUNT_COL,
        use_velocity=True,
    )
    instance = X.iloc[row_index].fillna(0)
    feature_names = list(X.columns)
    if background is None:
        background = X.fillna(0).head(min(500, len(X)))

    def predict_proba_np(arr: np.ndarray) -> np.ndarray:
        batch = pd.DataFrame(arr, columns=feature_names)
        return model.predict_proba(batch)

    explainer = LimeTabularExplainer(
        background.values,
        feature_names=feature_names,
        mode="classification",
        discretize_continuous=True,
    )
    exp = explainer.explain_instance(
        instance.values,
        predict_proba_np,
        num_features=min(12, len(feature_names)),
    )
    out = out_path or str(CFG.MODELS_DIR / "lime_explanation.html")
    exp.save_to_file(out)
    print(f"Saved LIME report: {out}")


if __name__ == "__main__":
    sample = os.environ.get("FRAUD_EXPLAIN_CSV", str(CFG.DATA_DIR / "sample_to_explain.csv"))
    row = int(os.environ.get("FRAUD_EXPLAIN_ROW", "0"))
    mode = os.environ.get("FRAUD_EXPLAIN_MODE", "lime").lower()
    if mode == "shap":
        explain_shap_tree(sample, row_index=row)
    else:
        explain_lime(sample, row_index=row)
