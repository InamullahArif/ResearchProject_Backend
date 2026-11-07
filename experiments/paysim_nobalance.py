"""
PaySim sensitivity analysis: balance information withheld.

On full PaySim features, fraud is almost perfectly separable through the ratio of
amount to the sender's pre-transaction balance (fraud typically drains the account),
which leaves no room for behavioural or anomaly components to add value.
This analysis removes all balance-derived inputs and asks whether receiver-level
behavioural features, and the hybrid score, then improve detection.

Same split protocol as paysim_experiments.py (stratified 70/15/15, seeds 0-4).
Usage: python paysim_nobalance.py PS_log.csv out_dir
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paysim_experiments import build_features  # noqa: E402
from run_experiments import IForestScorer, best_f1_threshold, hgb, metrics, rank01  # noqa: E402

NB_RAW = ["step", "is_transfer", "amount", "hour_of_day", "log_amount"]
BEH = ["dest_prior_count", "dest_count_24h", "dest_amount_deviation", "dest_first_seen"]
KEEP = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr")


def hybrid(yva, Pv, Pt, Av, At, Dv, Dt):
    best = None
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    for a, b in product(grid, grid):
        g = round(1 - a - b, 2)
        if g < -1e-9:
            continue
        g = max(g, 0.0)
        ap = average_precision_score(yva, a * Pv + b * Av + g * Dv)
        if best is None or ap > best[0]:
            best = (ap, a, b, g)
    _, a, b, g = best
    Rv, Rt = a * Pv + b * Av + g * Dv, a * Pt + b * At + g * Dt
    return Rv, Rt, [a, b, g]


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    df = build_features(src)
    y = df["isFraud"].to_numpy()
    allres = []
    for seed in range(5):
        idx = np.arange(len(df))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
        ytr, yva, yte = y[tr], y[va], y[te]
        res = {"seed": seed}
        D_all = df["dest_amount_deviation"].to_numpy()
        Dv_raw, Dt_raw = D_all[va], D_all[te]
        Dv, Dt = rank01(Dv_raw, Dv_raw), rank01(Dv_raw, Dt_raw)
        for tag, cols in [("no_behaviour", NB_RAW), ("with_behaviour", NB_RAW + BEH)]:
            X = df[cols].to_numpy(dtype=float)
            m = hgb(seed, "balanced").fit(X[tr], ytr)
            Pv, Pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
            sup = metrics(yte, Pt, best_f1_threshold(yva, Pv))
            iso = IForestScorer(seed).fit(X[tr][ytr == 0])
            Av_raw, At_raw = iso.score(X[va]), iso.score(X[te])
            Av, At = rank01(Av_raw, Av_raw), rank01(Av_raw, At_raw)
            Rv, Rt, w = hybrid(yva, Pv, Pt, Av, At, Dv, Dt)
            hy = metrics(yte, Rt, best_f1_threshold(yva, Rv))
            res[tag] = {"supervised": {k: sup[k] for k in KEEP},
                        "isoforest": {k: v for k, v in metrics(yte, At_raw, best_f1_threshold(yva, Av_raw)).items() if k in KEEP},
                        "hybrid": {k: hy[k] for k in KEEP} | {"weights": w}}
            print(seed, tag, "sup PR=%.4f F1=%.4f | hybrid PR=%.4f F1=%.4f w=%s" %
                  (sup["pr_auc"], sup["f1"], hy["pr_auc"], hy["f1"], w), flush=True)
        allres.append(res)
        (out / "nobalance_results.json").write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
