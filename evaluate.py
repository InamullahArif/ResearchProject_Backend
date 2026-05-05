# evaluate.py
import runtime_env  # noqa: F401

import argparse
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import CFG
from features import engineer_features
from model_io import load_joblib


def anomaly_scores_hybrid(bundle: dict, X: pd.DataFrame, Xn: pd.DataFrame) -> np.ndarray:
    """Blend Isolation Forest and autoencoder into [0,1] scores (vectorised)."""
    iso_m = bundle["isoforest"]
    raw_iso = -iso_m.score_samples(Xn)
    iso_p = 1.0 / (1.0 + np.exp(-raw_iso))
    ae_m = bundle.get("autoencoder")
    if ae_m is None:
        return iso_p
    z = ae_m.score_samples(X)
    ae_p = 1.0 / (1.0 + np.exp(-z))
    w1 = float(bundle.get("anomaly_blend_iso", CFG.ANOMALY_BLEND_ISO))
    w2 = float(bundle.get("anomaly_blend_ae", CFG.ANOMALY_BLEND_AE))
    s = w1 + w2
    if s <= 0:
        return iso_p
    return (w1 / s) * iso_p + (w2 / s) * ae_p


def load_model():
    path = CFG.MODELS_DIR / "fraud_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train first: python train.py"
        )
    return load_joblib(path, what="fraud_model.joblib")


def load_bundle():
    path = CFG.MODELS_DIR / "scoring_bundle.joblib"
    if not path.exists():
        return None
    return load_joblib(path, what="scoring_bundle.joblib")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = engineer_features(
        df,
        CFG.TIME_COL,
        CFG.USER_COL,
        CFG.AMOUNT_COL,
        use_velocity=True,
    )

    if CFG.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{CFG.TARGET_COL}' not found in CSV.")

    y = df[CFG.TARGET_COL].astype(int).to_numpy()
    X = df.drop(columns=[CFG.TARGET_COL])
    return X, y


def benchmark_latency(model, X: pd.DataFrame, n_samples: int = 5000) -> dict:
    n = min(len(X), n_samples)
    if n == 0:
        return {"n": 0, "avg_ms": None, "p95_ms": None, "tps": None}

    X_small = X.head(n)

    times = []
    for i in range(n):
        row = X_small.iloc[[i]]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    avg_ms = float(times.mean())
    p95_ms = float(np.percentile(times, 95))
    tps = float(1000.0 / avg_ms) if avg_ms > 0 else None

    return {"n": n, "avg_ms": avg_ms, "p95_ms": p95_ms, "tps": tps}


def evaluate(csv_path: str, threshold: float | None = None):
    threshold = CFG.ALERT_THRESHOLD if threshold is None else float(threshold)

    model = load_model()
    bundle = load_bundle()
    df = pd.read_csv(csv_path)

    X, y = prepare_features(df)

    probs = model.predict_proba(X)[:, 1]

    if bundle is not None:
        numeric_cols = bundle["numeric_cols"]
        Xn = X[[c for c in numeric_cols if c in X.columns]].fillna(0.0)
        anomaly_scores = anomaly_scores_hybrid(bundle, X, Xn)
        probs = (
            CFG.HYBRID_WEIGHT_ML * probs
            + CFG.HYBRID_WEIGHT_ANOMALY * anomaly_scores
        )

    preds = (probs >= threshold).astype(int)

    acc = float(accuracy_score(y, preds))
    pr_auc = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn + 1e-9))

    print("\n=== Evaluation Results ===")
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {acc:.6f}")
    print(f"PR-AUC:    {pr_auc:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"FPR:       {fpr:.6f}")

    print("\nConfusion Matrix [ [TN FP] [FN TP] ]")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y, preds, digits=4))

    lat = benchmark_latency(model, X, n_samples=2000)
    print("\n=== Latency Benchmark (single-row predict_proba) ===")
    print(f"Samples: {lat['n']}")
    print(f"Avg ms:  {lat['avg_ms']}")
    print(f"P95 ms:  {lat['p95_ms']}")
    print(f"TPS:     {lat['tps']}")

    return {
        "threshold": threshold,
        "accuracy": acc,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
        "latency": lat,
    }


def evaluate_all_models(
    csv_path: str,
    threshold: float | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Compute Accuracy, PR-AUC, ROC-AUC, F1 for every supervised model (incl. tab_transformer)
    and Isolation Forest on the given labelled CSV. Requires scoring_bundle.joblib from train.py.
    """
    threshold = CFG.ALERT_THRESHOLD if threshold is None else float(threshold)
    bundle = load_bundle()
    if bundle is None or not bundle.get("supervised_models"):
        raise FileNotFoundError(
            "scoring_bundle.joblib with supervised_models not found. Run: python train.py"
        )

    df = pd.read_csv(csv_path)
    X, y = prepare_features(df)
    y_test = y

    rows: list[dict] = []
    for name, est in bundle["supervised_models"].items():
        probs = est.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        rows.append(
            {
                "model": name,
                "Accuracy": float(accuracy_score(y_test, preds)),
                "PR_AUC": float(average_precision_score(y_test, probs)),
                "ROC_AUC": float(roc_auc_score(y_test, probs)),
                "F1": float(f1_score(y_test, preds, zero_division=0)),
            }
        )

    anomaly_model = bundle["isoforest"]
    numeric_cols = bundle["numeric_cols"]
    Xn = X[[c for c in numeric_cols if c in X.columns]].fillna(0.0)
    raw = -anomaly_model.score_samples(Xn)
    iso_scores = raw.copy()
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    iso_preds = (iso_scores >= threshold).astype(int)
    rows.append(
        {
            "model": "isoforest",
            "Accuracy": float(accuracy_score(y_test, iso_preds)),
            "PR_AUC": float(average_precision_score(y_test, iso_scores)),
            "ROC_AUC": float(roc_auc_score(y_test, iso_scores)),
            "F1": float(f1_score(y_test, iso_preds, zero_division=0)),
        }
    )

    ae_model = bundle.get("autoencoder")
    if ae_model is not None:
        ae_raw = ae_model.score_samples(X)
        ae_scores = (ae_raw - ae_raw.min()) / (ae_raw.max() - ae_raw.min() + 1e-9)
        ae_preds = (ae_scores >= threshold).astype(int)
        rows.append(
            {
                "model": "autoencoder",
                "Accuracy": float(accuracy_score(y_test, ae_preds)),
                "PR_AUC": float(average_precision_score(y_test, ae_scores)),
                "ROC_AUC": float(roc_auc_score(y_test, ae_scores)),
                "F1": float(f1_score(y_test, ae_preds, zero_division=0)),
            }
        )

    out = pd.DataFrame(rows)
    print("\n=== All models — same threshold for hard predictions ===")
    print(f"CSV: {csv_path}  |  threshold: {threshold}")
    print(out.to_string(index=False))
    if save_csv:
        path = CFG.MODELS_DIR / "evaluate_all_models_metrics.csv"
        out.to_csv(path, index=False)
        print(f"\nSaved: {path}")
    return out


def fairness_amount_quartiles(csv_path: str, threshold: float | None = None) -> pd.DataFrame:
    """Simple fairness-style slice: metrics per transaction-amount quartile."""
    threshold = CFG.ALERT_THRESHOLD if threshold is None else float(threshold)
    model = load_model()
    bundle = load_bundle()
    df = pd.read_csv(csv_path)
    X, y = prepare_features(df)
    if CFG.AMOUNT_COL not in X.columns:
        raise ValueError(f"Column '{CFG.AMOUNT_COL}' required for amount-based slices.")
    probs = model.predict_proba(X)[:, 1]
    if bundle is not None:
        Xn = X[[c for c in bundle["numeric_cols"] if c in X.columns]].fillna(0.0)
        probs = CFG.HYBRID_WEIGHT_ML * probs + CFG.HYBRID_WEIGHT_ANOMALY * anomaly_scores_hybrid(
            bundle, X, Xn
        )
    quart = pd.qcut(X[CFG.AMOUNT_COL].rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    rows = []
    for label in quart.cat.categories:
        mask = (quart == label).to_numpy()
        if mask.sum() == 0:
            continue
        preds = (probs[mask] >= threshold).astype(int)
        yy = y[mask]
        rows.append(
            {
                "slice": str(label),
                "n": int(mask.sum()),
                "fraud_rate": float(yy.mean()),
                "Accuracy": float(accuracy_score(yy, preds)),
                "PR_AUC": float(average_precision_score(yy, probs[mask])),
                "F1": float(f1_score(yy, preds, zero_division=0)),
            }
        )
    out = pd.DataFrame(rows)
    print("\n=== Fairness-style slices (amount quartiles) ===")
    print(out.to_string(index=False))
    out_path = CFG.MODELS_DIR / "fairness_amount_quartiles.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out


def cost_sensitive_best_threshold(csv_path: str) -> dict:
    """Grid-search threshold minimising FN/FP economic-style cost (proposal alignment)."""
    model = load_model()
    bundle = load_bundle()
    df = pd.read_csv(csv_path)
    X, y = prepare_features(df)
    probs = model.predict_proba(X)[:, 1]
    if bundle is not None:
        Xn = X[[c for c in bundle["numeric_cols"] if c in X.columns]].fillna(0.0)
        probs = CFG.HYBRID_WEIGHT_ML * probs + CFG.HYBRID_WEIGHT_ANOMALY * anomaly_scores_hybrid(
            bundle, X, Xn
        )
    c_fn = CFG.COST_FALSE_NEGATIVE
    c_fp = CFG.COST_FALSE_POSITIVE
    best = {"cost": float("inf"), "threshold": 0.5, "fn": 0, "fp": 0}
    for t in np.linspace(0.05, 0.95, 37):
        pred = (probs >= t).astype(int)
        cm = confusion_matrix(y, pred)
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        cost = fn * c_fn + fp * c_fp
        if cost < best["cost"]:
            best = {"cost": float(cost), "threshold": float(t), "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn)}
    print("\n=== Cost-sensitive threshold (hybrid score) ===")
    print(f"COST_FALSE_NEGATIVE={c_fn}, COST_FALSE_POSITIVE={c_fp}")
    print(best)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fraud models on a labelled CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(CFG.DATA_DIR / "test.csv"),
        help="Labelled test CSV (default: data/test.csv)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Print Accuracy/PR-AUC/ROC-AUC/F1 for every model including tab_transformer",
    )
    parser.add_argument(
        "--fairness",
        action="store_true",
        help="Report metrics by transaction amount quartile (simple slice analysis)",
    )
    parser.add_argument(
        "--cost-threshold",
        action="store_true",
        help="Search threshold minimising CFG.COST_FALSE_NEGATIVE*FN + CFG.COST_FALSE_POSITIVE*FP",
    )
    args = parser.parse_args()
    if args.fairness:
        fairness_amount_quartiles(args.csv_path)
    elif args.cost_threshold:
        cost_sensitive_best_threshold(args.csv_path)
    elif args.all_models:
        evaluate_all_models(args.csv_path)
    else:
        evaluate(args.csv_path)