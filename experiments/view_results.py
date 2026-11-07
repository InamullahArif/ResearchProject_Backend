"""
View experiment results as tables and charts.

Usage:
    python experiments/view_results.py results/creditcard/results.json
    python experiments/view_results.py results/paysim/results.json

Prints the main tables to the console and saves, next to the results file:
    model_comparison.csv, ablation.csv, imbalance.csv   (open in Excel)
    pr_auc_by_model.png, ablation.png                    (charts)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "latency_ms", "throughput_tps"]


def summarise(seeds, section, metrics):
    rows = []
    for name in seeds[0][section]:
        row = {"model": name}
        for m in metrics:
            vals = [s[section][name][m] for s in seeds if m in s[section][name]]
            if vals:
                row[m] = np.mean(vals)
                row[m + "_sd"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def show(df, title, metrics):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    out = pd.DataFrame({"model": df["model"]})
    for m in metrics:
        if m in df:
            if m == "fpr":
                out["FPR %"] = [f"{100 * a:.3f}" for a in df[m]]
            elif m == "throughput_tps":
                out["txn/s"] = [f"{a:,.0f}" for a in df[m]]
            elif m == "latency_ms":
                out["ms/txn"] = [f"{a:.2f}" for a in df[m]]
            else:
                out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[m], df[m + "_sd"])]
    print(out.to_string(index=False))


def main():
    path = Path(sys.argv[1])
    r = json.load(open(path))
    seeds = r["seeds"]
    out_dir = path.parent
    print(f"Results file: {path}   ({len(seeds)} split(s))")
    if "meta" in r:
        print("Dataset info:", r["meta"])
    print("Split sizes [rows, fraud]:", seeds[0]["split"])

    models = summarise(seeds, "models", METRICS)
    show(models, "MODEL COMPARISON (test set, mean ± sd over splits)", METRICS)
    models.to_csv(out_dir / "model_comparison.csv", index=False)

    w = [s["models"]["Proposed Hybrid Framework"].get("weights") for s in seeds]
    print("\nHybrid weights chosen on validation (alpha, beta, gamma) per split:", w)

    for sec, title in [("ablation", "ABLATION STUDY"), ("imbalance", "CLASS IMBALANCE STRATEGIES")]:
        if sec in seeds[0]:
            df = summarise(seeds, sec, ["precision", "recall", "f1", "pr_auc", "fpr"])
            show(df, title, ["precision", "recall", "f1", "pr_auc", "fpr"])
            df.to_csv(out_dir / f"{sec}.csv", index=False)

    if "perm_importance" in seeds[0]:
        imp = {}
        for s in seeds:
            for f, m, _ in s["perm_importance"]:
                imp.setdefault(f, []).append(m)
        top = sorted(((f, np.mean(v)) for f, v in imp.items()), key=lambda t: -t[1])[:10]
        print("\nTOP FEATURES (permutation importance, mean drop in PR-AUC):")
        for f, v in top:
            print(f"  {f:28s} {v:.4f}")

    if "leakage_check" in seeds[0]:
        lk = np.mean([s["leakage_check"]["pr_auc"] for s in seeds])
        print(f"\nLeakage check (balance columns included): PR-AUC {lk:.3f}  <- not valid, shown for comparison only")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        d = models.sort_values("pr_auc")
        ax.barh(d["model"], d["pr_auc"], xerr=d["pr_auc_sd"], color="#4C72B0")
        ax.set_xlabel("PR-AUC (mean ± sd)")
        ax.set_xlim(0, 1)
        ax.set_title(f"Model comparison: {path.parent.name}")
        fig.tight_layout()
        fig.savefig(out_dir / "pr_auc_by_model.png", dpi=150)
        if "ablation" in seeds[0]:
            a = summarise(seeds, "ablation", ["pr_auc"])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(a["model"][::-1], a["pr_auc"][::-1], xerr=a["pr_auc_sd"][::-1], color="#55A868")
            ax.set_xlabel("PR-AUC (mean ± sd)")
            ax.set_xlim(0, 1)
            ax.set_title(f"Ablation: {path.parent.name}")
            fig.tight_layout()
            fig.savefig(out_dir / "ablation.png", dpi=150)
        print(f"\nCharts and CSV tables saved in: {out_dir}")
    except ImportError:
        print(f"\nCSV tables saved in: {out_dir}  (install matplotlib for charts)")


if __name__ == "__main__":
    main()
