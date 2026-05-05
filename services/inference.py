"""
Loads scoring bundle / fallback model and exposes hybrid fraud scoring utilities.
Shared by FastAPI routes and evaluation scripts.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CFG
from model_io import load_joblib

logger = logging.getLogger(__name__)

BUNDLE_PATH = CFG.MODELS_DIR / "scoring_bundle.joblib"

bundle: dict[str, Any] = {}
model: Any = None
anomaly_model: Any = None
autoencoder_model: Any = None
numeric_cols: list[str] = []
supervised_models: dict[str, Any] | None = None
best_model_name: str | None = None
results_metrics: dict[str, Any] | None = None
_blend_iso: float = CFG.ANOMALY_BLEND_ISO
_blend_ae: float = CFG.ANOMALY_BLEND_AE
tree_pipeline_for_shap: Any | None = None  # RF pipeline for TreeExplainer
feature_names_after_transform: list[str] | None = None
standalone_model_path: Path | None = None


def _resolve_standalone_model_path() -> Path:
    """Pickle/joblib supervised pipeline when no scoring bundle is used."""
    override = os.environ.get("FRAUD_MODEL_PATH", "").strip()
    if override:
        p = Path(override)
        chosen = p if p.is_absolute() else (CFG.MODELS_DIR / p)
        return chosen.resolve()
    primary = (CFG.MODELS_DIR / CFG.FRAUD_MODEL_FILE).resolve()
    if primary.exists():
        return primary
    legacy = (CFG.MODELS_DIR / "fraud_model.joblib").resolve()
    if legacy.exists():
        logger.warning(
            "Standalone model %s not found; loading legacy fraud_model.joblib. "
            "Set CFG.FRAUD_MODEL_FILE or FRAUD_MODEL_PATH to your artifact.",
            primary,
        )
        return legacy
    return primary


def load_scoring_artifacts() -> None:
    """Populate module-level state from disk (call on startup or after retrain)."""
    global bundle, model, anomaly_model, autoencoder_model, numeric_cols
    global supervised_models, best_model_name, results_metrics
    global _blend_iso, _blend_ae, tree_pipeline_for_shap, feature_names_after_transform
    global standalone_model_path

    tree_pipeline_for_shap = None
    feature_names_after_transform = None
    standalone_model_path = None

    use_bundle = bool(getattr(CFG, "USE_SCORING_BUNDLE", True)) and BUNDLE_PATH.exists()
    if use_bundle:
        bundle = load_joblib(BUNDLE_PATH, what="scoring_bundle.joblib")
        model = bundle["best_model"]
        anomaly_model = bundle["isoforest"]
        autoencoder_model = bundle.get("autoencoder")
        numeric_cols = list(bundle.get("numeric_cols", []))
        supervised_models = bundle.get("supervised_models")
        best_model_name = bundle.get("best_model_name")
        results_metrics = bundle.get("results_metrics")
        _blend_iso = float(bundle.get("anomaly_blend_iso", CFG.ANOMALY_BLEND_ISO))
        _blend_ae = float(bundle.get("anomaly_blend_ae", CFG.ANOMALY_BLEND_AE))
        tree_pipeline_for_shap = bundle.get("tree_pipeline_for_shap")
        feature_names_after_transform = bundle.get("shap_feature_names")
    else:
        bundle = {}
        standalone_model_path = _resolve_standalone_model_path()
        model = load_joblib(standalone_model_path, what=standalone_model_path.name)
        anomaly_model = None
        autoencoder_model = None
        numeric_cols = []
        supervised_models = None
        best_model_name = standalone_model_path.stem
        results_metrics = None
        _blend_iso = CFG.ANOMALY_BLEND_ISO
        _blend_ae = CFG.ANOMALY_BLEND_AE

    logger.info(
        "Scoring artifacts loaded: bundle=%s best_model=%s artifact=%s",
        bool(bundle),
        best_model_name,
        str(standalone_model_path) if standalone_model_path else BUNDLE_PATH.name,
    )


load_scoring_artifacts()


def loaded_model_artifact() -> str | None:
    """Filesystem path of the active bundle or standalone pipeline (for health / predict metadata)."""
    if bundle:
        return str(BUNDLE_PATH)
    if standalone_model_path is not None:
        return str(standalone_model_path)
    return None


def risk_level(score: float) -> str:
    if score >= CFG.HIGH_RISK_THRESHOLD:
        return "high"
    if score >= CFG.MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def anomaly_score_row(X_last: pd.DataFrame) -> float:
    Xn = X_last[[c for c in numeric_cols if c in X_last.columns]].fillna(0.0)
    iso_p = 0.0
    if anomaly_model is not None and len(Xn.columns) > 0:
        r = -float(anomaly_model.score_samples(Xn)[0])
        iso_p = float(1.0 / (1.0 + np.exp(-r)))
    ae_p = 0.0
    if autoencoder_model is not None:
        z = float(autoencoder_model.score_samples(X_last)[0])
        ae_p = float(1.0 / (1.0 + np.exp(-z)))
    if autoencoder_model is not None and anomaly_model is not None:
        s = _blend_iso + _blend_ae
        if s <= 0:
            return iso_p
        return float((_blend_iso / s) * iso_p + (_blend_ae / s) * ae_p)
    if autoencoder_model is not None:
        return ae_p
    return iso_p


def anomaly_scores_batch(X: pd.DataFrame, Xn: pd.DataFrame | None = None) -> np.ndarray:
    """Vectorised hybrid anomaly scores in [0, 1]."""
    if Xn is None:
        Xn = X[[c for c in numeric_cols if c in X.columns]].fillna(0.0)
    iso_p = np.zeros(len(X), dtype=np.float64)
    if anomaly_model is not None and Xn.shape[1] > 0:
        raw = -anomaly_model.score_samples(Xn)
        iso_p = 1.0 / (1.0 + np.exp(-raw))
    ae_p = np.zeros(len(X), dtype=np.float64)
    if autoencoder_model is not None:
        z = autoencoder_model.score_samples(X)
        ae_p = 1.0 / (1.0 + np.exp(-z))
    if autoencoder_model is not None and anomaly_model is not None:
        s = _blend_iso + _blend_ae
        if s <= 0:
            return iso_p
        return (_blend_iso / s) * iso_p + (_blend_ae / s) * ae_p
    if autoencoder_model is not None:
        return ae_p
    return iso_p


def hybrid_scores_ml_anomaly(X: pd.DataFrame, ml_probs: np.ndarray) -> np.ndarray:
    """Combine supervised probabilities with anomaly detector (same weights as /score)."""
    Xn = X[[c for c in numeric_cols if c in X.columns]].fillna(0.0)
    an = anomaly_scores_batch(X, Xn)
    return CFG.HYBRID_WEIGHT_ML * ml_probs + CFG.HYBRID_WEIGHT_ANOMALY * an


def build_reasons(
    last_row: pd.Series,
    ml_score: float,
    anomaly_score: float,
    hybrid_score: float,
) -> list[str]:
    reasons: list[str] = []
    if float(last_row.get("rolling_tx_count_1h", 0.0)) >= 4:
        reasons.append("High transaction velocity in last hour")
    if float(last_row.get("amount_zscore_24h", 0.0)) >= 2.5:
        reasons.append("Transaction amount is unusual vs recent behavior")
    if ml_score >= CFG.ALERT_THRESHOLD:
        reasons.append("Supervised fraud model confidence is above alert threshold")
    if anomaly_score >= 0.70:
        reasons.append("Anomaly detector flags this transaction as unusual")
    if not reasons and hybrid_score >= CFG.MEDIUM_RISK_THRESHOLD:
        reasons.append("Combined score indicates elevated risk")
    return reasons
