from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from config import CFG
from model_io import load_joblib


def main():
    report_path = CFG.MODELS_DIR / "train_report.joblib"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Training report not found at {report_path}. "
            "Run train.py first to generate it."
        )

    payload = load_joblib(report_path, what="train_report.joblib")
    results: dict = payload["results"]
    eval_data: dict = payload.get("eval_data")

    if eval_data is None:
        raise ValueError(
            "No eval_data found in train_report.joblib. "
            "Please rerun train.py to regenerate the report with evaluation data."
        )

    y_test = np.asarray(eval_data["y_test"])
    supervised_probs: dict = eval_data["supervised"]
    iso_scores = np.asarray(eval_data["iso_scores"])

    model_names = list(supervised_probs.keys()) + ["isoforest"]

    # ----- ROC curves -----
    plt.figure(figsize=(7, 6))
    for name, probs in supervised_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = results[name]["ROC_AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_scores)
    auc_iso = results["isoforest"]["ROC_AUC"]
    plt.plot(fpr_iso, tpr_iso, label=f"isoforest (AUC={auc_iso:.3f})", linestyle="--")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    roc_path = CFG.MODELS_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    print(f"Saved ROC curves figure to: {roc_path}")

    # ----- Precision-Recall curves -----
    plt.figure(figsize=(7, 6))
    for name, probs in supervised_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = results[name]["PR_AUC"]
        plt.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={pr_auc:.3f})",
        )

    precision_iso, recall_iso, _ = precision_recall_curve(y_test, iso_scores)
    pr_auc_iso = results["isoforest"]["PR_AUC"]
    plt.plot(
        recall_iso,
        precision_iso,
        label=f"isoforest (PR-AUC={pr_auc_iso:.3f})",
        linestyle="--",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    pr_path = CFG.MODELS_DIR / "pr_curves.png"
    plt.savefig(pr_path, dpi=150)
    print(f"Saved PR curves figure to: {pr_path}")


if __name__ == "__main__":
    main()

