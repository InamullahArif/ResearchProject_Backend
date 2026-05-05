"""
Production FastAPI surface: dataset upload, batch/single prediction, analytics,
legacy /score, optional streaming queue metrics.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

import runtime_env  # noqa: F401

from api.deps import verify_api_key
from config import CFG
from dataset_loaders import load_dataset
from pipeline.feature_engineering import engineer_features
from services import inference, transaction_history
from services.dataset_store import PredictionBatchResult, store
from services.explain_service import shap_values_for_row, tree_pipeline_feature_importance
from services.prediction_queue import get_or_create_queue

logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str
    user_id: str
    amount: float
    merchant: str | None = None
    category: str | None = None
    location: str | None = None
    device_type: str | None = None
    channel: str | None = None


class PredictSingleResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Hybrid score 0–1")
    label: str = Field(..., description="fraud or legitimate (threshold-based)")
    ml_probability: float | None = None
    anomaly_score: float | None = None
    risk_level: str
    model_artifact: str | None = Field(
        default=None,
        description="Path of scoring_bundle.joblib or standalone model (e.g. .pkl) used for ML probability.",
    )
    feature_importance: dict[str, float] = Field(default_factory=dict)
    shap_explanations: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float | None = None


def _ensure_v_cols(row: dict[str, Any]) -> None:
    for i in range(1, 29):
        key = f"V{i}"
        row[key] = row.get(key, 0.0)


def _score_last_from_history() -> tuple[pd.DataFrame, float, float, float]:
    df = transaction_history.as_dataframe()
    df = engineer_features(
        df,
        CFG.TIME_COL,
        CFG.USER_COL,
        CFG.AMOUNT_COL,
        use_velocity=True,
    )
    X_last = df.tail(1)
    t0 = time.perf_counter()
    ml_p = float(inference.model.predict_proba(X_last)[0, 1])
    an = inference.anomaly_score_row(X_last)
    hyb = float(
        CFG.HYBRID_WEIGHT_ML * ml_p + CFG.HYBRID_WEIGHT_ANOMALY * an,
    )
    _ = time.perf_counter() - t0
    return X_last, ml_p, an, hyb


def create_app() -> FastAPI:
    application = FastAPI(
        title="Financial Fraud Detection API",
        description=(
            "Transformer + classical ensembles + anomaly (Isolation Forest, Autoencoder). "
            "Endpoints for CSV upload, batch metrics, analytics, and real-time scoring."
        ),
        version="2.0.0",
    )

    # Browsers send OPTIONS preflight for cross-origin POST + custom headers (e.g. X-API-Key).
    # Default allowlist matches a typical React/Next dev server on port 3000.
    _cors_default = "http://localhost:3000,http://127.0.0.1:3000"
    _cors = os.environ.get("CORS_ALLOW_ORIGINS", _cors_default).strip()
    if _cors == "*":
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        origins = [o.strip() for o in _cors.split(",") if o.strip()]
        if not origins:
            origins = [o.strip() for o in _cors_default.split(",")]
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @application.get("/health")
    def health():
        return {
            "status": "ok",
            "best_model": inference.best_model_name,
            "bundle_loaded": bool(inference.bundle),
            "model_artifact": inference.loaded_model_artifact(),
            "api_key_required": bool(os.environ.get("FRAUD_API_KEY", "").strip()),
            "supervised_models_available": list(inference.supervised_models.keys())
            if inference.supervised_models
            else [],
        }

    @application.get("/metrics", dependencies=[Depends(verify_api_key)])
    def training_metrics():
        if not inference.results_metrics:
            return {
                "detail": "No results_metrics in scoring bundle. Run train.py first.",
            }
        return {
            "best_model": inference.best_model_name,
            "results": inference.results_metrics,
        }

    @application.post("/score", dependencies=[Depends(verify_api_key)])
    def score_tx(
        tx: Transaction,
        breakdown: bool = Query(
            False,
            description="If true, include fraud probabilities from each supervised model.",
        ),
    ):
        row = tx.model_dump()
        _ensure_v_cols(row)
        transaction_history.append(row)
        X_last, ml_proba, anomaly_score, hybrid_score = _score_last_from_history()
        risk_level = inference.risk_level(hybrid_score)
        is_alert = bool(hybrid_score >= CFG.ALERT_THRESHOLD)
        reasons = inference.build_reasons(
            X_last.iloc[0], ml_proba, anomaly_score, hybrid_score
        )
        out: dict[str, Any] = {
            "fraud_probability": hybrid_score,
            "ml_probability": ml_proba,
            "best_supervised_model": inference.best_model_name,
            "anomaly_score": anomaly_score,
            "risk_level": risk_level,
            "alert": is_alert,
            "threshold": CFG.ALERT_THRESHOLD,
            "risk_bands": {
                "low": f"< {CFG.MEDIUM_RISK_THRESHOLD}",
                "medium": f"{CFG.MEDIUM_RISK_THRESHOLD} - {CFG.HIGH_RISK_THRESHOLD}",
                "high": f">= {CFG.HIGH_RISK_THRESHOLD}",
            },
            "reasons": reasons,
        }
        if breakdown and inference.supervised_models:
            per_model: dict[str, float] = {}
            for name, est in inference.supervised_models.items():
                try:
                    per_model[name] = float(est.predict_proba(X_last)[0, 1])
                except Exception:
                    per_model[name] = float("nan")
            out["supervised_breakdown"] = per_model
        return out

    @application.post("/upload-dataset")
    async def upload_dataset(
        file: UploadFile = File(..., description="Transaction CSV"),
        preset: str = Form("generic"),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="CSV must be UTF-8") from exc
        preset_l = (preset or "generic").lower().strip()
        if preset_l == "generic":
            df = pd.read_csv(io.StringIO(text))
        else:
            tmp_path = CFG.DATA_DIR / "_upload_tmp.csv"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(text, encoding="utf-8")
            df = load_dataset(str(tmp_path), preset_l)
        has_target = CFG.TARGET_COL in df.columns
        try:
            df_eng = engineer_features(
                df,
                CFG.TIME_COL,
                CFG.USER_COL,
                CFG.AMOUNT_COL,
                use_velocity=True,
            )
        except Exception as exc:
            logger.exception("Feature engineering failed")
            raise HTTPException(
                status_code=422,
                detail=f"Preprocessing failed: {exc!s}",
            ) from exc
        uid = store.set_upload(df_eng, has_target=has_target)
        return {
            "upload_id": uid,
            "rows": int(len(df_eng)),
            "columns": list(df_eng.columns),
            "has_target": has_target,
            "preset": preset_l,
        }

    @application.post("/predict", dependencies=[Depends(verify_api_key)])
    def predict_batch(threshold: float | None = Query(None, description="Override alert threshold")):
        sess = store.get_session()
        if sess is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded. POST /upload-dataset first.")
        thr = float(threshold) if threshold is not None else float(CFG.ALERT_THRESHOLD)
        df = sess.df
        target_col = CFG.TARGET_COL
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
        else:
            X = df
        t0 = time.perf_counter()
        ml_probs = inference.model.predict_proba(X)[:, 1]
        hybrid = inference.hybrid_scores_ml_anomaly(X, ml_probs)
        dt = (time.perf_counter() - t0) * 1000.0 / max(len(X), 1)
        fraud_flags = (hybrid >= thr).astype(int)
        an_scores = inference.anomaly_scores_batch(X)
        store.set_prediction_result(
            PredictionBatchResult(
                hybrid_scores=hybrid,
                ml_scores=ml_probs,
                anomaly_scores=an_scores,
                fraud_flags=fraud_flags,
                threshold=thr,
            )
        )
        n = len(hybrid)
        fraud_count = int(fraud_flags.sum())
        return {
            "total_transactions": n,
            "fraud_count": fraud_count,
            "fraud_percentage": float(100.0 * fraud_count / n) if n else 0.0,
            "average_risk_score": float(np.mean(hybrid)) if n else 0.0,
            "threshold": thr,
            "avg_latency_ms_per_row": float(dt),
            "model_artifact": inference.loaded_model_artifact(),
            "best_model": inference.best_model_name,
        }

    @application.get("/analytics", dependencies=[Depends(verify_api_key)])
    def analytics(
        top_k: int = Query(20, ge=1, le=200),
        bins: int = Query(20, ge=5, le=100),
    ):
        sess = store.get_session()
        pred = store.get_last_prediction()
        if sess is None or pred is None:
            raise HTTPException(
                status_code=400,
                detail="Upload a dataset and run POST /predict first.",
            )
        df = sess.df
        scores = pred.hybrid_scores
        thr = pred.threshold

        hist, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
        risk_histogram = {
            "bins": int(bins),
            "counts": hist.tolist(),
            "edges": edges.tolist(),
        }

        fraud_dist: dict[str, Any]
        target_col = CFG.TARGET_COL
        if target_col in df.columns and len(df) == len(scores):
            y = df[target_col].astype(int).to_numpy()
            pred_f = (scores >= thr).astype(int)
            fraud_dist = {
                "ground_truth_fraud_rate": float(y.mean()),
                "predicted_fraud_rate": float(pred_f.mean()),
                "labelled_rows": int(len(y)),
            }
        else:
            pred_f = (scores >= thr).astype(int)
            fraud_dist = {
                "predicted_fraud_count": int(pred_f.sum()),
                "predicted_fraud_rate": float(pred_f.mean()),
                "note": "No ground-truth column in upload",
            }

        order = np.argsort(-scores)
        display_cols = [
            c
            for c in (
                CFG.TIME_COL,
                CFG.USER_COL,
                CFG.AMOUNT_COL,
                "merchant",
                "channel",
            )
            if c in df.columns
        ]
        top_idx = order[:top_k]
        suspicious: list[dict[str, Any]] = []
        for i in top_idx:
            row = {"risk_score": float(scores[i]), "row_index": int(i)}
            for c in display_cols:
                v = df.iloc[int(i)][c]
                row[c] = v if pd.notna(v) else None
            suspicious.append(row)

        extra: dict[str, Any] = {}
        if inference.tree_pipeline_for_shap is not None:
            extra["model_feature_importance"] = tree_pipeline_feature_importance(
                inference.tree_pipeline_for_shap
            )

        return {
            "fraud_distribution": fraud_dist,
            "risk_score_histogram": risk_histogram,
            "top_suspicious_transactions": suspicious,
            **extra,
        }

    @application.post("/predict-single", dependencies=[Depends(verify_api_key)])
    def predict_single(
        tx: Transaction,
        explain: bool = Query(False, description="Include SHAP values (tree model)"),
    ):
        t0 = time.perf_counter()
        row = tx.model_dump()
        _ensure_v_cols(row)
        transaction_history.append(row)
        X_last, ml_p, an, hyb = _score_last_from_history()
        label = "fraud" if hyb >= CFG.ALERT_THRESHOLD else "legitimate"
        shap_list: list[dict[str, Any]] = []
        importance: dict[str, float] = {}
        if explain and inference.tree_pipeline_for_shap is not None:
            dfh = transaction_history.as_dataframe()
            full_eng = engineer_features(
                dfh,
                CFG.TIME_COL,
                CFG.USER_COL,
                CFG.AMOUNT_COL,
                use_velocity=True,
            )
            bg_eng = full_eng.iloc[:-1].tail(500) if len(full_eng) > 1 else None
            shap_list, importance = shap_values_for_row(
                inference.tree_pipeline_for_shap,
                X_last,
                bg_eng,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictSingleResponse(
            fraud_probability=float(hyb),
            label=label,
            ml_probability=float(ml_p),
            anomaly_score=float(an),
            risk_level=inference.risk_level(hyb),
            model_artifact=inference.loaded_model_artifact(),
            feature_importance=importance,
            shap_explanations=shap_list,
            latency_ms=round(latency_ms, 3),
        )

    @application.post("/reload-models", dependencies=[Depends(verify_api_key)])
    def reload_models():
        """Reload joblib artifacts after retraining (same process)."""
        inference.load_scoring_artifacts()
        return {
            "status": "reloaded",
            "best_model": inference.best_model_name,
            "model_artifact": inference.loaded_model_artifact(),
        }

    @application.get("/queue-stats", dependencies=[Depends(verify_api_key)])
    def queue_stats():
        q = get_or_create_queue()
        s = q.stats()
        return {
            "queue_depth": s.depth,
            "processed_total": s.processed_total,
            "avg_latency_ms": s.avg_latency_ms,
        }

    return application


app = create_app()
