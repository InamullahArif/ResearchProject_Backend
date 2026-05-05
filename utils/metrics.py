"""Consolidated classification metrics for fraud evaluation and API responses."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    denom = fp + tn
    fpr = float(fp / denom) if denom > 0 else 0.0

    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else 0.0,
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def latency_stats_ms(model: Any, X: pd.DataFrame, *, n_samples: int = 500) -> dict[str, Any]:
    """Per-row predict_proba latency (ms) for production monitoring."""
    n = min(len(X), max(1, n_samples))
    X_small = X.head(n)
    times: list[float] = []
    for i in range(n):
        row = X_small.iloc[[i]]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times, dtype=np.float64)
    return {
        "n": n,
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }
