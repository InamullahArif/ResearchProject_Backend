"""
PaySim experiments for the manuscript (second dataset).

Data: PaySim synthetic mobile-money log (6,362,620 transactions, 8,213 fraud).
Fraud occurs only in TRANSFER and CASH_OUT, so the evaluation uses those two types.

Behavioural features are built per *receiving* account (nameDest), because 99.85%
of sending accounts (nameOrig) appear only once. All behavioural features are
computed causally: each transaction only sees earlier transactions (by step, then
file order) across the full log, and they use no labels.

All four balance columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest,
newbalanceDest) are excluded from the main feature set, as the dataset's documentation
instructs: fraudulent transactions were cancelled in the simulation, so the balances
leak the label. A separate check reports how much they inflate performance.

Usage: python paysim_experiments.py PS_log.csv out_dir [n_seeds]
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import (AutoencoderScorer, IForestScorer, best_f1_threshold, hgb,
                             metrics, rank01, smote, time_batch, time_single)

EPS = 1e-6
RAW = ["step", "is_transfer", "amount"]
ENG = ["hour_of_day", "log_amount",
       "dest_prior_count", "dest_count_24h", "dest_amount_deviation", "dest_first_seen"]
LEAKY = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "amount_to_orig_balance"]


def build_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order"] = np.arange(len(df))
    # ---- receiver-level behaviour over the full log (all types), causal
    dest_code = pd.factorize(df["nameDest"])[0].astype(np.int64)
    key = dest_code * 1000 + df["step"].to_numpy()          # step <= 743
    o = np.lexsort((df["order"].to_numpy(), df["step"].to_numpy(), dest_code))
    k_sorted = key[o]
    d_sorted = dest_code[o]
    pos = np.arange(len(o))
    first = np.r_[0, np.flatnonzero(np.diff(d_sorted)) + 1]
    grp_start = np.repeat(first, np.diff(np.r_[first, len(o)]))
    prior = pos - grp_start
    lower = np.searchsorted(k_sorted, d_sorted * 1000 + np.maximum(df["step"].to_numpy()[o] - 24, 0), side="left")
    lower = np.maximum(lower, grp_start)
    cnt24 = pos - lower
    la = np.log1p(df["amount"].to_numpy())[o]
    s1 = np.cumsum(la)
    s2 = np.cumsum(la ** 2)
    # sums over prior rows within group
    base1 = np.where(grp_start > 0, s1[grp_start - 1], 0.0)
    base2 = np.where(grp_start > 0, s2[grp_start - 1], 0.0)
    ps1 = s1 - la - base1
    ps2 = s2 - la ** 2 - base2
    n = prior.astype(float)
    mean = np.divide(ps1, n, out=np.zeros_like(ps1), where=n > 0)
    var = np.divide(ps2, n, out=np.zeros_like(ps2), where=n > 0) - mean ** 2
    sd = np.sqrt(np.clip(var, 0, None))
    dev = np.where(n >= 2, np.abs(la - mean) / (sd + 0.1), 0.0)   # 0.1 floor on log-scale sd
    feats = np.empty((len(o), 4))
    feats[o, 0] = prior
    feats[o, 1] = cnt24
    feats[o, 2] = dev
    feats[o, 3] = (prior == 0)
    df["dest_prior_count"], df["dest_count_24h"], df["dest_amount_deviation"], df["dest_first_seen"] = feats.T
    # ---- transaction-level
    df["hour_of_day"] = df["step"] % 24
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    return df


def supervised_models(seed):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_samples=0.25,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "Gradient Boosting (HistGB)": hgb(seed, "balanced"),
    }


def run_seed(df, seed, ae_sub=300_000):
    rng = np.random.default_rng(seed)
    y = df["isFraud"].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    cols = RAW + ENG
    X = df[cols].to_numpy(dtype=float)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    res = {"seed": seed, "split": {k: [int(len(v)), int(y[v].sum())] for k, v in dict(train=tr, val=va, test=te).items()}}
    rows, sva_d, ste_d, fitted = {}, {}, {}, {}
    nlat = 300

    for name, m in supervised_models(seed).items():
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
        sva, ste = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        r = metrics(yte, ste, best_f1_threshold(yva, sva))
        r["latency_ms"] = time_single(lambda q, m=m: m.predict_proba(q), Xte, nlat)
        r["throughput_tps"] = time_batch(lambda A, m=m: m.predict_proba(A), Xte)
        r["fit_s"] = fit_s
        rows[name], sva_d[name], ste_d[name], fitted[name] = r, sva, ste, m
        print(f"   {name} fit {fit_s:.0f}s PR={r['pr_auc']:.3f}", flush=True)

    legit = Xtr[ytr == 0]
    sub = rng.choice(len(legit), min(ae_sub, len(legit)), replace=False)
    for name, m, fit_X in [("Isolation Forest", IForestScorer(seed), legit),
                           ("Autoencoder", AutoencoderScorer(seed), legit[sub])]:
        t0 = time.perf_counter(); m.fit(fit_X); fit_s = time.perf_counter() - t0
        if isinstance(m, IForestScorer):
            m.m.n_jobs = 1
        sva, ste = m.score(Xva), m.score(Xte)
        r = metrics(yte, ste, best_f1_threshold(yva, sva))
        r["latency_ms"] = time_single(lambda q, m=m: m.score(q), Xte, nlat)
        r["throughput_tps"] = time_batch(lambda A, m=m: m.score(A), Xte)
        r["fit_s"] = fit_s
        rows[name], sva_d[name], ste_d[name], fitted[name] = r, sva, ste, m
        print(f"   {name} fit {fit_s:.0f}s PR={r['pr_auc']:.3f}", flush=True)

    # ---- hybrid: P from HistGB, A from Isolation Forest, D = receiver amount deviation
    sup_name = "Gradient Boosting (HistGB)"
    sup, iso = fitted[sup_name], fitted["Isolation Forest"]
    ci = cols.index("dest_amount_deviation")
    P_va, P_te = sva_d[sup_name], ste_d[sup_name]
    Aref, Dref = sva_d["Isolation Forest"], Xva[:, ci]
    A_va, A_te = rank01(Aref, Aref), rank01(Aref, ste_d["Isolation Forest"])
    D_va, D_te = rank01(Dref, Dref), rank01(Dref, Xte[:, ci])
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)

    def select(allow_d=True):
        best, curve = None, []
        for a, b in product(grid, grid):
            g = round(1 - a - b, 2)
            if g < -1e-9 or (not allow_d and g > 1e-9):
                continue
            ap = average_precision_score(yva, a * P_va + b * A_va + max(g, 0) * D_va)
            curve.append((a, b, max(g, 0.0), ap))
            if best is None or ap > best[0]:
                best = (ap, a, b, max(g, 0.0))
        return best[1:], curve

    (a, b, g), curve = select(True)
    R_va, R_te = a * P_va + b * A_va + g * D_va, a * P_te + b * A_te + g * D_te
    T2 = best_f1_threshold(yva, R_va)
    T1 = min(float(np.quantile(R_va[yva == 1], 0.05)), T2)
    r = metrics(yte, R_te, T2)
    As, Ds = np.sort(Aref), np.sort(Dref)

    def pipe(q):
        p = sup.predict_proba(q)[:, 1]
        an = np.searchsorted(As, iso.score(q), side="right") / len(As)
        dv = np.searchsorted(Ds, q[:, ci], side="right") / len(Ds)
        return a * p + b * an + g * dv
    r["latency_ms"] = time_single(pipe, Xte, nlat)
    r["throughput_tps"] = time_batch(pipe, Xte)
    r["weights"], r["T1"], r["T2"] = [a, b, g], T1, T2
    band = np.where(R_te >= T2, "alert", np.where(R_te >= T1, "monitor", "accept"))
    r["bands"] = {k: {"n": int((band == k).sum()), "fraud": int(yte[band == k].sum())} for k in ["accept", "monitor", "alert"]}
    rows["Proposed Hybrid Framework"] = r
    # validation PR-AUC of a few fixed weightings, to show sensitivity
    cdict = {(c[0], c[1], c[2]): c[3] for c in curve}
    res["val_prauc_weightings"] = {f"{k}": cdict.get(k) for k in [(1.0, 0.0, 0.0), (0.8, 0.2, 0.0), (0.8, 0.0, 0.2), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2)]}
    res["models"] = rows

    # ---- ablation (HistGB)
    raw_idx = [cols.index(c) for c in RAW]
    no_beh = [cols.index(c) for c in RAW + ["hour_of_day", "log_amount"]]

    def ev(m, Xa, ya, ci_):
        m.fit(Xa, ya)
        sv, st = m.predict_proba(Xva[:, ci_])[:, 1], m.predict_proba(Xte[:, ci_])[:, 1]
        return metrics(yte, st, best_f1_threshold(yva, sv))
    full = list(range(len(cols)))
    abl = {}
    abl["Baseline (raw features, no imbalance handling)"] = ev(hgb(seed, None), Xtr[:, raw_idx], ytr, raw_idx)
    abl["+ Transaction-level features"] = ev(hgb(seed, None), Xtr[:, no_beh], ytr, no_beh)
    abl["+ Receiver behavioural features"] = ev(hgb(seed, None), Xtr, ytr, full)
    abl["+ Imbalance handling (cost-sensitive)"] = rows[sup_name]
    (a2, b2, _), _ = select(False)
    s2v, s2t = a2 * P_va + b2 * A_va, a2 * P_te + b2 * A_te
    abl["+ Anomaly score"] = metrics(yte, s2t, best_f1_threshold(yva, s2v)) | {"weights": [a2, b2, 0.0]}
    abl["Full hybrid framework (+ behavioural deviation)"] = r
    keep = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "weights")
    res["ablation"] = {k: {kk: vv for kk, vv in v.items() if kk in keep} for k, v in abl.items()}

    # ---- imbalance strategies (HistGB, all features)
    imb = {"No handling": abl["+ Receiver behavioural features"]}
    maj = np.flatnonzero(ytr == 0)
    kp = np.concatenate([rng.choice(maj, size=(ytr == 1).sum(), replace=False), np.flatnonzero(ytr == 1)])
    imb["Random under-sampling (1:1)"] = ev(hgb(seed, None), Xtr[kp], ytr[kp], full)
    sc = StandardScaler().fit(Xtr)
    Xs, ys = smote(sc.transform(Xtr), ytr, rng, ratio=0.1)
    imb["SMOTE over-sampling (1:10)"] = ev(hgb(seed, None), sc.inverse_transform(Xs), ys, full)
    imb["Cost-sensitive learning (balanced weights)"] = rows[sup_name]
    res["imbalance"] = {k: {kk: vv for kk, vv in v.items() if kk in keep} for k, v in imb.items()}

    # ---- leakage check: add post-transaction balances
    lk_cols = cols + LEAKY
    XL = df[lk_cols].to_numpy(dtype=float)
    mL = hgb(seed, "balanced").fit(XL[tr], ytr)
    res["leakage_check"] = metrics(yte, mL.predict_proba(XL[te])[:, 1],
                                   best_f1_threshold(yva, mL.predict_proba(XL[va])[:, 1]))

    # ---- errors
    pred = R_te >= T2
    fnm, tpm, fpm = (~pred) & (yte == 1), pred & (yte == 1), pred & (yte == 0)
    res["errors"] = {"fn": int(fnm.sum()), "tp": int(tpm.sum()), "fp": int(fpm.sum()),
                     "fn_in_monitor_band": int(((R_te >= T1) & fnm).sum()),
                     "fn_transfer_share": float(Xte[fnm, cols.index("is_transfer")].mean()) if fnm.any() else None,
                     "fn_dest_first_seen_share": float(Xte[fnm, cols.index("dest_first_seen")].mean()) if fnm.any() else None,
                     "tp_dest_first_seen_share": float(Xte[tpm, cols.index("dest_first_seen")].mean()),
                     "legit_dest_first_seen_share": float(Xte[yte == 0, cols.index("dest_first_seen")].mean())}

    # ---- permutation importance
    s_idx = np.concatenate([np.flatnonzero(yva == 1), rng.choice(np.flatnonzero(yva == 0), min(40000, int((yva == 0).sum())), replace=False)])
    pi = permutation_importance(sup, Xva[s_idx], yva[s_idx], scoring="average_precision", n_repeats=5, random_state=seed)
    order = np.argsort(-pi.importances_mean)
    res["perm_importance"] = [[cols[i], float(pi.importances_mean[i]), float(pi.importances_std[i])] for i in order]
    return res


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = build_features(src)
    meta = {"rows_all_types": 6362620, "rows_transfer_cashout": len(df), "fraud": int(df["isFraud"].sum()),
            "orig_unique_share": None, "feature_build_s": time.time() - t0,
            "fraud_dest_first_seen_share": float(df.loc[df.isFraud == 1, "dest_first_seen"].mean()),
            "legit_dest_first_seen_share": float(df.loc[df.isFraud == 0, "dest_first_seen"].mean()),
            "fraud_median_dest_count24": float(df.loc[df.isFraud == 1, "dest_count_24h"].median()),
            "legit_median_dest_count24": float(df.loc[df.isFraud == 0, "dest_count_24h"].median())}
    print(meta, flush=True)
    allres = {"meta": meta, "seeds": []}
    for s in range(n_seeds):
        t1 = time.time()
        r = run_seed(df, s)
        allres["seeds"].append(r)
        print(f"seed {s} done in {time.time() - t1:.0f}s", flush=True)
        for k, v in r["models"].items():
            print(f"  {k:30s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} ROC={v['roc_auc']:.4f} "
                  f"PR={v['pr_auc']:.3f} FPR={v['fpr']:.5f} lat={v['latency_ms']:.2f}ms tps={v['throughput_tps']:.0f}", flush=True)
        print("  weights", r["models"]["Proposed Hybrid Framework"]["weights"], r["val_prauc_weightings"], flush=True)
        print("  leakage", {k: round(v, 4) for k, v in r["leakage_check"].items() if isinstance(v, float)}, flush=True)
        (out / "results.json").write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
