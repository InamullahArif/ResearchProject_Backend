from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CFG
from model_io import load_joblib


def main():
    report_path = CFG.MODELS_DIR / "train_report.joblib"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Training report not found at {report_path}. "
            "Run train.py first to generate it."
        )

    report = load_joblib(report_path, what="train_report.joblib")
    results: dict = report["results"]

    model_names = list(results.keys())
    metrics = ["Accuracy", "PR_AUC", "ROC_AUC", "F1"]

    # Prepare data (older train_report may lack Accuracy)
    values = {
        m: [results[name].get(m, float("nan")) for name in model_names]
        for m in metrics
    }

    x = np.arange(len(model_names))
    width = 0.2

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    fig.suptitle("Model Performance Comparison", fontsize=14)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x, values[metric], width, color="tab:blue")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        for i, v in enumerate(values[metric]):
            if np.isnan(v):
                continue
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = CFG.MODELS_DIR / "metrics_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved metrics comparison figure to: {out_path}")


if __name__ == "__main__":
    main()

