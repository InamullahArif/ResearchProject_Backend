"""
Combine all experiment result files into results/summary.json, which the API serves at
GET /research/summary and the frontend "Research Results" page displays.

Usage (from the backend project folder):
    python experiments/build_summary.py results
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "latency_ms", "throughput_tps"]
LABELS = {"creditcard": "European credit card (ULB)", "paysim": "PaySim mobile money"}


def ms(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return [float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0]


def rows(runs, metrics):
    """runs: list of {name: {metric: value}} -> list of {name, metric: [mean, sd]}"""
    out = []
    for name in runs[0]:
        r = {"name": name}
        for m in metrics:
            v = ms([run[name].get(m) for run in runs if m in run[name]])
            if v is not None:
                r[m] = v
        out.append(r)
    return out


def load(p):
    return json.load(open(p)) if p.exists() else None


def main_results(r):
    S = r["seeds"]
    d = {"split": S[0]["split"], "n_splits": len(S),
         "models": rows([s["models"] for s in S], METRICS),
         "hybrid_weights": [s["models"]["Proposed Hybrid Framework"].get("weights") for s in S]}
    for sec in ["ablation", "imbalance"]:
        if sec in S[0]:
            d[sec] = rows([s[sec] for s in S], ["precision", "recall", "f1", "pr_auc", "fpr"])
    if "perm_importance" in S[0]:
        imp = {}
        for s in S:
            for f, m, _ in s["perm_importance"]:
                imp.setdefault(f, []).append(m)
        d["importance"] = [{"feature": f, "value": float(np.mean(v))}
                           for f, v in sorted(imp.items(), key=lambda t: -np.mean(t[1]))[:10]]
    if "leakage_check" in S[0]:
        d["leakage_pr_auc"] = ms([s["leakage_check"]["pr_auc"] for s in S])
    if "meta" in r:
        d["meta"] = r["meta"]
    bands = [s["models"]["Proposed Hybrid Framework"].get("bands") for s in S]
    if bands[0]:
        d["bands"] = bands
    return d


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}
    for ds in ["creditcard", "paysim"]:
        r = load(root / ds / "results.json")
        if r is None:
            continue
        d = {"label": LABELS[ds]} | main_results(r)
        if ds == "creditcard":
            d["transactions"] = r["meta"]["rows_dedup"]
            d["fraud"] = r["meta"]["fraud_dedup"]
        else:
            d["transactions"] = r["meta"]["rows_transfer_cashout"]
            d["fraud"] = r["meta"]["fraud"]
            nb = load(root / ds / "nobalance_results.json")
            if nb:
                d["behaviour_cost_sensitive"] = {
                    tag: ms([s[tag]["supervised"]["pr_auc"] for s in nb]) for tag in ["no_behaviour", "with_behaviour"]}
        summary["datasets"][ds] = d

    ext = root / "extended"
    t = load(ext / "temporal_results.json")
    if t:
        for ds, v in t.items():
            if ds in summary["datasets"]:
                summary["datasets"][ds]["temporal"] = {"split": v["info"]["split"], "fraud_rate": v["info"]["fraud_rate"],
                                                       "models": rows(v["runs"], ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]),
                                                       "hybrid_weights": [run["Proposed Hybrid Framework"]["weights"] for run in v["runs"]]}
    a = load(ext / "anomaly_feature_results.json")
    if a:
        for ds, v in a.items():
            if ds in summary["datasets"]:
                summary["datasets"][ds]["anomaly_feature"] = {
                    "class_weight": v["class_weight"],
                    "variants": rows(v["runs"], ["precision", "recall", "f1", "pr_auc", "fpr"]),
                    "pr_auc_diff_vs_gb": {k: [float(run[k]["pr_auc"] - run["GB"]["pr_auc"]) for run in v["runs"]]
                                          for k in v["runs"][0] if k != "GB"}}
    tu = load(ext / "tuning_results.json")
    if tu:
        for ds, v in tu.items():
            if ds in summary["datasets"]:
                summary["datasets"][ds]["tuning"] = {
                    "best_params": [run["best_params"] for run in v["runs"]],
                    "tuned": rows([{"t": run["tuned"]} for run in v["runs"]], ["precision", "recall", "f1", "pr_auc", "fpr"])[0],
                    "default": rows([{"d": run["default"]} for run in v["runs"]], ["precision", "recall", "f1", "pr_auc", "fpr"])[0]}
    out = root / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"Wrote {out} with datasets: {list(summary['datasets'])}")


if __name__ == "__main__":
    main()
