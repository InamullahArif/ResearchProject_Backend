"""In-memory store for uploaded CSVs and last batch prediction results (analytics)."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class UploadSession:
    """Holds engineered features and optional ground-truth labels."""

    upload_id: str
    df: pd.DataFrame
    has_target: bool
    created_at: str = ""
    raw_row_count: int = 0


@dataclass
class PredictionBatchResult:
    """Cached results for GET /analytics."""

    hybrid_scores: np.ndarray
    ml_scores: np.ndarray
    anomaly_scores: np.ndarray
    fraud_flags: np.ndarray
    threshold: float


@dataclass
class DatasetStore:
    """Thread-safe singleton for demo / single-node deployments."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _session: UploadSession | None = None
    _last_pred: PredictionBatchResult | None = None

    def set_upload(self, df: pd.DataFrame, *, has_target: bool) -> str:
        from datetime import datetime, timezone

        uid = str(uuid.uuid4())
        with self._lock:
            self._session = UploadSession(
                upload_id=uid,
                df=df,
                has_target=has_target,
                created_at=datetime.now(timezone.utc).isoformat(),
                raw_row_count=len(df),
            )
            self._last_pred = None
        return uid

    def get_session(self) -> UploadSession | None:
        with self._lock:
            return self._session

    def clear(self) -> None:
        with self._lock:
            self._session = None
            self._last_pred = None

    def set_prediction_result(self, result: PredictionBatchResult) -> None:
        with self._lock:
            self._last_pred = result

    def get_last_prediction(self) -> PredictionBatchResult | None:
        with self._lock:
            return self._last_pred


store = DatasetStore()
