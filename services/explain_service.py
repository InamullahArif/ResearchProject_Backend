"""
SHAP explanations (tree models) and feature importance for API responses.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _strip_calibrator(model: Any) -> Any:
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        return model.calibrated_classifiers_[0].estimator
    return model


def shap_values_for_row(
    tree_pipeline: Any,
    X_row: pd.DataFrame,
    _background_X: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Returns (ranked shap contributions, aggregate feature_importance from RF if available).
    Uses TreeExplainer on the tree classifier after ColumnTransformer preprocessing.
    """
    base = _strip_calibrator(tree_pipeline)
    if not hasattr(base, "named_steps"):
        return [], {}
    pre = base.named_steps.get("pre")
    clf = base.named_steps.get("clf")
    if pre is None or clf is None:
        return [], {}

    importance: dict[str, float] = {}
    if hasattr(clf, "feature_importances_"):
        try:
            fn = np.asarray(pre.get_feature_names_out())
            imp = clf.feature_importances_
            importance = {
                str(fn[i]): float(imp[i]) for i in range(min(len(fn), len(imp)))
            }
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25])
        except Exception:
            importance = {}

    try:
        X_trans = pre.transform(X_row)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans, dtype=np.float64)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_trans)
        if isinstance(sv, list):
            v = np.asarray(sv[1] if len(sv) > 1 else sv[0])
        else:
            v = np.asarray(sv)
        if v.ndim > 1:
            v = v[0]

        fn = getattr(pre, "get_feature_names_out", None)
        names = list(fn()) if callable(fn) else [f"f{i}" for i in range(len(v))]
        pairs = sorted(
            [{"feature": str(names[i]), "shap_value": float(v[i])} for i in range(min(len(names), len(v)))],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )[:15]
        return pairs, importance
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return [], importance


def tree_pipeline_feature_importance(tree_pipeline: Any) -> dict[str, float]:
    """Mean decrease impurity from the tree classifier inside a sklearn Pipeline."""
    base = _strip_calibrator(tree_pipeline)
    if not hasattr(base, "named_steps"):
        return {}
    pre = base.named_steps.get("pre")
    clf = base.named_steps.get("clf")
    if pre is None or clf is None or not hasattr(clf, "feature_importances_"):
        return {}
    try:
        fn = np.asarray(pre.get_feature_names_out())
        imp = clf.feature_importances_
        out = {
            str(fn[i]): float(imp[i]) for i in range(min(len(fn), len(imp)))
        }
        return dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:30])
    except Exception:
        return {}
