"""
Extended experiments for the manuscript (robustness checks requested for a PhD-standard study).

  temporal  - time-ordered 70/15/15 split (train on the past, test on the future) on both
              datasets; model randomness repeated over 3 seeds.
  anomfeat  - anomaly scores (Isolation Forest, autoencoder) used as extra INPUT features to
              gradient boosting, instead of being combined with its output (5 random splits).
  tuning    - validation-based hyperparameter search for gradient boosting (5 random splits).

Usage:
  python extended_experiments.py temporal|anomfeat|tuning creditcard.csv paysim.csv out_dir
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paysim_experiments import ENG as PS_ENG, RAW as PS_RAW, build_features  # noqa: E402
from run_experiments import (ENG_COLS, RAW_COLS, AutoencoderScorer, DeviationProfile,  # noqa: E402
                             IForestScorer, add_engineered, best_f1_threshold, hgb, metrics, rank01)

KEEP = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr")
PS_TXN = PS_RAW + ["hour_of_day", "log_amount"]


# ------------------------------------------------------------------ data
def load_cc(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return df


def cc_matrix(df, tr):
    prof = DeviationProfile().fit(df.iloc[tr].loc[df.iloc[tr]["Class"] == 0, "Amount"].to_numpy())
    f = add_engineered(df, prof)
    cols = RAW_COLS + ENG_COLS
    return f[cols].to_numpy(float), df["Class"].to_numpy(), cols


def load_ps(path):
    return build_features(path)


def ps_matrix(df):
    cols = PS_RAW + PS_ENG
    return df[cols].to_numpy(float), df["isFraud"].to_numpy(), cols


def random_split(y, seed):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    return tr, va, te


def temporal_split(order):
    n = len(order)
    a, b = int(0.70 * n), int(0.85 * n)
    return order[:a], order[a:b], order[b:]


def ev(m, X, y, tr, va, te, cols_idx=None):
    ci = slice(None) if cols_idx is None else cols_idx
    m.fit(X[tr][:, ci], y[tr])
    sv = m.predict_proba(X[va][:, ci])[:, 1]
    st = m.predict_proba(X[te][:, ci])[:, 1]
    r = metrics(y[te], st, best_f1_threshold(y[va], sv))
    return {k: r[k] for k in KEEP}, sv, st


# ------------------------------------------------------------------ temporal
def temporal(cc_path, ps_path, out):
    res = {}
    for name in ["creditcard", "paysim"]:
        if name == "creditcard":
            df = load_cc(cc_path)
            order = np.argsort(df["Time"].to_numpy(), kind="stable")
            tr, va, te = temporal_split(order)
            X, y, cols = cc_matrix(df, tr)
            dev_col = cols.index("amount_deviation")
        else:
            df = load_ps(ps_path)
            order = np.lexsort((df["order"].to_numpy(), df["step"].to_numpy()))
            tr, va, te = temporal_split(order)
            X, y, cols = ps_matrix(df)
            dev_col = cols.index("dest_amount_deviation")
        info = {"split": {k: [int(len(v)), int(y[v].sum())] for k, v in dict(train=tr, val=va, test=te).items()},
                "fraud_rate": {k: float(y[v].mean()) for k, v in dict(train=tr, val=va, test=te).items()}}
        print(name, info, flush=True)
        runs = []
        for seed in range(3):
            t0 = time.time()
            r = {}
            models = {
                "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed),
                "Random Forest": RandomForestClassifier(n_estimators=200 if name == "creditcard" else 100, min_samples_leaf=2,
                                                        max_samples=None if name == "creditcard" else 0.25,
                                                        class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
                "Gradient Boosting (cost-sensitive)": hgb(seed, "balanced"),
                "Gradient Boosting (no imbalance handling)": hgb(seed, None),
            }
            sc = {}
            for mn, m in models.items():
                r[mn], sv, st = ev(m, X, y, tr, va, te)
                sc[mn] = (sv, st)
            legit = X[tr][y[tr] == 0]
            rng = np.random.default_rng(seed)
            sub = legit[rng.choice(len(legit), min(300_000, len(legit)), replace=False)]
            for mn, m, fx in [("Isolation Forest", IForestScorer(seed), legit), ("Autoencoder", AutoencoderScorer(seed), sub)]:
                m.fit(fx)
                sv, st = m.score(X[va]), m.score(X[te])
                rr = metrics(y[te], st, best_f1_threshold(y[va], sv))
                r[mn] = {k: rr[k] for k in KEEP}
                sc[mn] = (sv, st)
            # hybrid on the cost-sensitive GB (as in the main protocol)
            Pv, Pt = sc["Gradient Boosting (cost-sensitive)"]
            Av_raw, At_raw = sc["Isolation Forest"]
            Av, At = rank01(Av_raw, Av_raw), rank01(Av_raw, At_raw)
            Dv, Dt = rank01(X[va][:, dev_col], X[va][:, dev_col]), rank01(X[va][:, dev_col], X[te][:, dev_col])
            best = None
            g_ = np.round(np.arange(0, 1.0001, 0.05), 2)
            for a, b in product(g_, g_):
                g = round(1 - a - b, 2)
                if g < -1e-9:
                    continue
                g = max(g, 0.0)
                ap = average_precision_score(y[va], a * Pv + b * Av + g * Dv)
                if best is None or ap > best[0]:
                    best = (ap, a, b, g)
            _, a, b, g = best
            hr = metrics(y[te], a * Pt + b * At + g * Dt, best_f1_threshold(y[va], a * Pv + b * Av + g * Dv))
            r["Proposed Hybrid Framework"] = {k: hr[k] for k in KEEP} | {"weights": [a, b, g]}
            if name == "paysim":
                txn_idx = [cols.index(c) for c in PS_TXN]
                r["GB no handling, transaction features only"] = ev(hgb(seed, None), X, y, tr, va, te, txn_idx)[0]
            runs.append(r)
            print(f"  {name} seed {seed} {time.time() - t0:.0f}s  " +
                  "  ".join(f"{k[:22]}={v['pr_auc']:.3f}" for k, v in r.items()), flush=True)
        res[name] = {"info": info, "runs": runs}
        (out / "temporal_results.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ anomaly scores as features
def anomfeat(cc_path, ps_path, out):
    res = {}
    for name in ["creditcard", "paysim"]:
        df = load_cc(cc_path) if name == "creditcard" else load_ps(ps_path)
        y_all = df["Class"].to_numpy() if name == "creditcard" else df["isFraud"].to_numpy()
        cw = "balanced" if name == "creditcard" else None   # best imbalance setting per dataset
        runs = []
        for seed in range(5):
            t0 = time.time()
            tr, va, te = random_split(y_all, seed)
            X, y, cols = cc_matrix(df, tr) if name == "creditcard" else ps_matrix(df)
            legit = X[tr][y[tr] == 0]
            rng = np.random.default_rng(seed)
            iso = IForestScorer(seed).fit(legit)
            ae = AutoencoderScorer(seed).fit(legit[rng.choice(len(legit), min(300_000, len(legit)), replace=False)])
            A = iso.score(X)[:, None]
            R = np.log1p(ae.score(X))[:, None]
            variants = {"GB": X, "GB + Isolation Forest score": np.hstack([X, A]),
                        "GB + autoencoder error": np.hstack([X, R]), "GB + both anomaly scores": np.hstack([X, A, R])}
            r = {k: ev(hgb(seed, cw), V, y, tr, va, te)[0] for k, V in variants.items()}
            runs.append(r)
            print(f"  {name} seed {seed} {time.time() - t0:.0f}s  " +
                  "  ".join(f"{k}={v['pr_auc']:.4f}" for k, v in r.items()), flush=True)
        res[name] = {"class_weight": cw, "runs": runs}
        (out / "anomaly_feature_results.json").write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ tuning
GRID = list(product([0.05, 0.1], [15, 31, 63], [None, "balanced"]))


def tuning(cc_path, ps_path, out):
    res = {}
    for name in ["creditcard", "paysim"]:
        df = load_cc(cc_path) if name == "creditcard" else load_ps(ps_path)
        y_all = df["Class"].to_numpy() if name == "creditcard" else df["isFraud"].to_numpy()
        runs = []
        for seed in range(5):
            t0 = time.time()
            tr, va, te = random_split(y_all, seed)
            X, y, cols = cc_matrix(df, tr) if name == "creditcard" else ps_matrix(df)
            trials = []
            for lr, leaves, cw in GRID:
                m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=400, max_leaf_nodes=leaves,
                                                   l2_regularization=1.0, class_weight=cw, early_stopping=False,
                                                   random_state=seed)
                m.fit(X[tr], y[tr])
                sv = m.predict_proba(X[va])[:, 1]
                trials.append((average_precision_score(y[va], sv), lr, leaves, cw, m, sv))
            best = max(trials, key=lambda t: t[0])
            _, lr, leaves, cw, m, sv = best
            st = m.predict_proba(X[te])[:, 1]
            rr = metrics(y[te], st, best_f1_threshold(y[va], sv))
            default = next(t for t in trials if t[1] == 0.05 and t[2] == 31 and t[3] == "balanced")
            dst = default[4].predict_proba(X[te])[:, 1]
            dr = metrics(y[te], dst, best_f1_threshold(y[va], default[5]))
            runs.append({"best_params": {"learning_rate": lr, "max_leaf_nodes": leaves, "class_weight": cw},
                         "val_pr_auc_grid": [[t[1], t[2], t[3], float(t[0])] for t in trials],
                         "tuned": {k: rr[k] for k in KEEP}, "default": {k: dr[k] for k in KEEP}})
            print(f"  {name} seed {seed} {time.time() - t0:.0f}s best={lr},{leaves},{cw} tuned PR={rr['pr_auc']:.4f} "
                  f"default PR={dr['pr_auc']:.4f}", flush=True)
        res[name] = {"grid": [list(g) for g in GRID], "runs": runs}
        (out / "tuning_results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    mode, cc, ps, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    {"temporal": temporal, "anomfeat": anomfeat, "tuning": tuning}[mode](cc, ps, out)
