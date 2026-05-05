import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from config import CFG
from model_io import load_joblib


def _load_eval_data():
    path = CFG.MODELS_DIR / "train_report.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    payload = load_joblib(path, what="train_report.joblib")
    eval_data = payload.get("eval_data")
    if eval_data is None:
        raise ValueError("No eval_data in train_report. Re-run train.py.")
    return eval_data


def _hybrid_scores(eval_data: dict) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(eval_data["y_test"])
    supervised = eval_data["supervised"]
    report_path = CFG.MODELS_DIR / "train_report.joblib"
    best = None
    if report_path.exists():
        payload = load_joblib(report_path, what="train_report.joblib")
        best = payload.get("best")
    if best and best in supervised:
        ml = np.asarray(supervised[best])
    elif "xgb" in supervised:
        ml = np.asarray(supervised["xgb"])
    else:
        ml = np.asarray(supervised[next(iter(supervised.keys()))])
    iso = np.asarray(eval_data["iso_scores"])
    hybrid = CFG.HYBRID_WEIGHT_ML * ml + CFG.HYBRID_WEIGHT_ANOMALY * iso
    return y, hybrid


def plot_threshold_tradeoff(y: np.ndarray, scores: np.ndarray):
    thresholds = np.linspace(0.05, 0.95, 37)
    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "fpr": fp / (fp + tn + 1e-9),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(CFG.MODELS_DIR / "threshold_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(out["threshold"], out["precision"], label="Precision")
    plt.plot(out["threshold"], out["recall"], label="Recall")
    plt.plot(out["threshold"], out["f1"], label="F1")
    plt.plot(out["threshold"], out["fpr"], label="FPR")
    plt.axvline(CFG.ALERT_THRESHOLD, color="k", linestyle="--", alpha=0.5, label="Alert threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold Trade-off (Hybrid Score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.MODELS_DIR / "threshold_tradeoff.png", dpi=150)
    print(f"Saved: {CFG.MODELS_DIR / 'threshold_tradeoff.png'}")
    print(f"Saved: {CFG.MODELS_DIR / 'threshold_metrics.csv'}")


def plot_confusion_heatmap(y: np.ndarray, scores: np.ndarray):
    pred = (scores >= CFG.ALERT_THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Hybrid Score)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Fraud"])
    plt.yticks([0, 1], ["Normal", "Fraud"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(CFG.MODELS_DIR / "confusion_matrix_hybrid.png", dpi=150)
    print(f"Saved: {CFG.MODELS_DIR / 'confusion_matrix_hybrid.png'}")


def plot_score_distribution(y: np.ndarray, scores: np.ndarray):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[y == 0], bins=60, alpha=0.6, label="Normal", density=True)
    plt.hist(scores[y == 1], bins=60, alpha=0.6, label="Fraud", density=True)
    plt.axvline(CFG.ALERT_THRESHOLD, color="k", linestyle="--", label="Alert threshold")
    plt.xlabel("Hybrid fraud score")
    plt.ylabel("Density")
    plt.title("Score Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CFG.MODELS_DIR / "score_distribution.png", dpi=150)
    print(f"Saved: {CFG.MODELS_DIR / 'score_distribution.png'}")


def main():
    eval_data = _load_eval_data()
    y, scores = _hybrid_scores(eval_data)
    plot_threshold_tradeoff(y, scores)
    plot_confusion_heatmap(y, scores)
    plot_score_distribution(y, scores)


if __name__ == "__main__":
    main()

