"""
Reproducible experiments for the manuscript
"A Machine Learning-Based Real-Time Financial Fraud Detection Framework".

Dataset: ULB/Kaggle Credit Card Fraud Detection (284,807 rows; Time, V1-V28, Amount, Class).
Protocol: duplicates removed; stratified 70/15/15 split; repeated over 5 random seeds.
Hyperparameters, hybrid weights and thresholds are chosen on the validation split;
the test split is used once per seed.

Only scikit-learn / NumPy / pandas / SciPy are required.
Usage: python run_experiments.py path/to/creditcard.csv out_dir
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEEDS = [0, 1, 2, 3, 4]
EPS = 1e-6
V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + V_COLS + ["Amount"]
ENG_COLS = ["hour_of_day", "log_amount", "amount_deviation"]


# ----------------------------------------------------------------- features
class DeviationProfile:
    """Amount profile learned from legitimate training transactions only.

    The ULB dataset has no customer identifier, so the behavioural profile is a
    population-level profile of legitimate spending (mean/std of log amount),
    not a per-customer profile."""

    def fit(self, amount: np.ndarray):
        la = np.log1p(amount)
        self.mu, self.sigma = la.mean(), la.std()
        return self

    def deviation(self, amount: np.ndarray) -> np.ndarray:
        return np.abs(np.log1p(amount) - self.mu) / (self.sigma + EPS)


def add_engineered(df: pd.DataFrame, prof: DeviationProfile) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"] = (out["Time"] / 3600.0) % 24
    out["log_amount"] = np.log1p(out["Amount"])
    out["amount_deviation"] = prof.deviation(out["Amount"].to_numpy())
    return out


# ----------------------------------------------------------------- helpers
def smote(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, k: int = 5,
          ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE (Chawla et al., 2002): interpolate between minority neighbours until
    minority:majority == ratio."""
    Xm = X[y == 1]
    n_new = int(ratio * (y == 0).sum()) - len(Xm)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
    idx = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new)
    nb = idx[base, rng.integers(0, k, n_new)]
    gap = rng.random((n_new, 1))
    Xs = Xm[base] + gap * (Xm[nb] - Xm[base])
    return np.vstack([X, Xs]), np.concatenate([y, np.ones(n_new, dtype=int)])


def rank01(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] using the empirical CDF of reference (validation) scores."""
    s = np.sort(ref)
    return np.searchsorted(s, x, side="right") / len(s)


def best_f1_threshold(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])


def metrics(y, s, thr) -> dict:
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return dict(precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                fpr=fp / (fp + tn), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def time_single(fn, X: np.ndarray, n: int = 300) -> float:
    """Median wall-clock time (ms) to score one transaction at a time."""
    ts = []
    for i in range(n):
        row = X[i:i + 1]
        t0 = time.perf_counter()
        fn(row)
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def time_batch(fn, X: np.ndarray) -> float:
    """Throughput (transactions per second) when scoring the whole test split."""
    t0 = time.perf_counter()
    fn(X)
    return len(X) / (time.perf_counter() - t0)


# ----------------------------------------------------------------- model zoo
def supervised_models(seed: int, cost_sensitive: bool = True):
    cw = "balanced" if cost_sensitive else None
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                class_weight=cw, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample" if cw else None,
            n_jobs=-1, random_state=seed),
        "Gradient Boosting (HistGB)": hgb(seed, cw),
    }


def hgb(seed, cw="balanced"):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight=cw, early_stopping=False, random_state=seed)


class AutoencoderScorer:
    """Under-complete autoencoder trained on legitimate transactions;
    anomaly score = squared reconstruction error."""

    def __init__(self, seed):
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu",
                                max_iter=40, batch_size=512, learning_rate_init=1e-3,
                                random_state=seed)

    def fit(self, X):
        Z = self.scaler.fit_transform(X)
        self.net.fit(Z, Z)
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        return ((Z - self.net.predict(Z)) ** 2).sum(axis=1)


class IForestScorer:
    def __init__(self, seed, contamination="auto"):
        self.m = IsolationForest(n_estimators=200, max_samples=256,
                                 contamination=contamination, random_state=seed, n_jobs=-1)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


# ----------------------------------------------------------------- one seed
def run_seed(df: pd.DataFrame, seed: int, out: Path, extras: bool) -> dict:
    rng = np.random.default_rng(seed)
    y_all = df["Class"].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y_all, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y_all[tmp], random_state=seed)

    prof = DeviationProfile().fit(df.iloc[tr].loc[df.iloc[tr]["Class"] == 0, "Amount"].to_numpy())
    feat = add_engineered(df, prof)
    cols_full = RAW_COLS + ENG_COLS
    X = feat[cols_full].to_numpy()
    Xraw = feat[RAW_COLS].to_numpy()
    y = y_all
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    res = {"seed": seed, "split": {k: [int(len(v)), int(y[v].sum())] for k, v in
                                   dict(train=tr, val=va, test=te).items()}}
    model_rows, scores_te, scores_va, fitted = {}, {}, {}, {}

    # supervised
    for name, m in supervised_models(seed).items():
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1  # score single-threaded, as in a per-request service
        sva, ste = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, sva)
        row = metrics(yte, ste, thr)
        row["latency_ms"] = time_single(lambda r, m=m: m.predict_proba(r), Xte)
        row["throughput_tps"] = time_batch(lambda A, m=m: m.predict_proba(A), Xte)
        row["fit_s"] = fit_s
        model_rows[name], scores_va[name], scores_te[name], fitted[name] = row, sva, ste, m

    # anomaly detectors: trained on legitimate training transactions only
    legit = Xtr[ytr == 0]
    for name, m in [("Isolation Forest", IForestScorer(seed)),
                    ("Autoencoder", AutoencoderScorer(seed))]:
        t0 = time.perf_counter()
        m.fit(legit)
        fit_s = time.perf_counter() - t0
        if isinstance(m, IForestScorer):
            m.m.n_jobs = 1
        sva, ste = m.score(Xva), m.score(Xte)
        thr = best_f1_threshold(yva, sva)
        row = metrics(yte, ste, thr)
        row["latency_ms"] = time_single(lambda r, m=m: m.score(r), Xte)
        row["throughput_tps"] = time_batch(lambda A, m=m: m.score(A), Xte)
        row["fit_s"] = fit_s
        model_rows[name], scores_va[name], scores_te[name], fitted[name] = row, sva, ste, m

    # hybrid risk score: Risk = a*P_sup + b*A + g*D, (a,b,g) on simplex, chosen on validation
    sup_name = "Gradient Boosting (HistGB)"
    sup, iso = fitted[sup_name], fitted["Isolation Forest"]
    ci = ENG_COLS.index("amount_deviation") + len(RAW_COLS)
    A_va_raw, A_te_raw = scores_va["Isolation Forest"], scores_te["Isolation Forest"]
    D_va_raw, D_te_raw = Xva[:, ci], Xte[:, ci]
    P_va, P_te = scores_va[sup_name], scores_te[sup_name]
    A_va, A_te = rank01(A_va_raw, A_va_raw), rank01(A_va_raw, A_te_raw)
    D_va, D_te = rank01(D_va_raw, D_va_raw), rank01(D_va_raw, D_te_raw)

    def hybrid_select(allow_d=True):
        best = None
        grid = np.round(np.arange(0, 1.0001, 0.05), 2)
        for a, b in product(grid, grid):
            g = round(1 - a - b, 2)
            if g < -1e-9 or (not allow_d and g > 1e-9):
                continue
            s = a * P_va + b * A_va + g * D_va
            ap = average_precision_score(yva, s)
            if best is None or ap > best[0]:
                best = (ap, a, b, max(g, 0.0))
        return best[1:]

    a, b, g = hybrid_select(True)
    R_va = a * P_va + b * A_va + g * D_va
    R_te = a * P_te + b * A_te + g * D_te
    T2 = best_f1_threshold(yva, R_va)
    # T1: lowest score that still keeps >= 95% validation fraud above it (monitor band)
    T1 = float(np.quantile(R_va[yva == 1], 0.05))
    T1 = min(T1, T2)
    row = metrics(yte, R_te, T2)

    va_A_sorted = np.sort(A_va_raw)
    va_D_sorted = np.sort(D_va_raw)

    def hybrid_pipeline(rows):
        p = sup.predict_proba(rows)[:, 1]
        an = np.searchsorted(va_A_sorted, iso.score(rows), side="right") / len(va_A_sorted)
        dv = np.searchsorted(va_D_sorted, rows[:, ci], side="right") / len(va_D_sorted)
        return a * p + b * an + g * dv

    row["latency_ms"] = time_single(hybrid_pipeline, Xte)
    row["throughput_tps"] = time_batch(hybrid_pipeline, Xte)
    row["weights"] = [a, b, g]
    row["T1"], row["T2"] = T1, T2
    band = np.where(R_te >= T2, "alert", np.where(R_te >= T1, "monitor", "accept"))
    row["bands"] = {k: {"n": int((band == k).sum()), "fraud": int(yte[band == k].sum())}
                    for k in ["accept", "monitor", "alert"]}
    model_rows["Proposed Hybrid Framework"] = row
    scores_te["Proposed Hybrid Framework"] = R_te
    res["models"] = model_rows

    # paired bootstrap: hybrid vs HistGB on the same test transactions
    B = 1000
    diffs_ap, diffs_f1 = [], []
    pred_h = (R_te >= T2).astype(int)
    pred_s = (P_te >= best_f1_threshold(yva, P_va)).astype(int)
    for _ in range(B):
        bi = rng.integers(0, len(yte), len(yte))
        yb = yte[bi]
        if yb.sum() == 0:
            continue
        diffs_ap.append(average_precision_score(yb, R_te[bi]) - average_precision_score(yb, P_te[bi]))

        def f1(pr):
            tp = (pr[bi] & yb).sum(); fp = (pr[bi] & (1 - yb)).sum(); fn = ((1 - pr[bi]) & yb).sum()
            return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        diffs_f1.append(f1(pred_h) - f1(pred_s))
    res["bootstrap_hybrid_minus_hgb"] = {
        "pr_auc": [float(np.mean(diffs_ap)), *np.percentile(diffs_ap, [2.5, 97.5]).tolist()],
        "f1": [float(np.mean(diffs_f1)), *np.percentile(diffs_f1, [2.5, 97.5]).tolist()],
    }

    if not extras:
        return res

    # ---------------- ablation (HistGB base learner)
    abl = {}
    raw_idx = list(range(len(RAW_COLS)))

    def eval_sup(m, Xa_tr, ya_tr, cols):
        m.fit(Xa_tr, ya_tr)
        sv, st = m.predict_proba(Xva[:, cols])[:, 1], m.predict_proba(Xte[:, cols])[:, 1]
        return metrics(yte, st, best_f1_threshold(yva, sv)), sv, st

    full_idx = list(range(X.shape[1]))
    abl["Baseline (raw features, no imbalance handling)"] = eval_sup(hgb(seed, None), Xtr[:, raw_idx], ytr, raw_idx)[0]
    abl["+ Feature engineering"] = eval_sup(hgb(seed, None), Xtr, ytr, full_idx)[0]
    abl["+ Imbalance handling (cost-sensitive)"] = model_rows[sup_name] | {}
    a2, b2, _ = hybrid_select(False)
    s2v, s2t = a2 * P_va + b2 * A_va, a2 * P_te + b2 * A_te
    abl["+ Anomaly score (supervised + anomaly)"] = metrics(yte, s2t, best_f1_threshold(yva, s2v)) | {"weights": [a2, b2, 0.0]}
    abl["Full hybrid framework (+ behavioural deviation)"] = row
    res["ablation"] = {k: {kk: vv for kk, vv in v.items() if kk in
                           ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "weights")}
                       for k, v in abl.items()}

    # ---------------- imbalance strategies (HistGB base learner, all features)
    imb = {}
    imb["No handling"] = abl["+ Feature engineering"]
    maj = np.flatnonzero(ytr == 0)
    keep = np.concatenate([rng.choice(maj, size=(ytr == 1).sum(), replace=False), np.flatnonzero(ytr == 1)])
    imb["Random under-sampling (1:1)"] = eval_sup(hgb(seed, None), Xtr[keep], ytr[keep], full_idx)[0]
    sc = StandardScaler().fit(Xtr)
    Xs, ys = smote(sc.transform(Xtr), ytr, rng, ratio=0.1)
    imb["SMOTE over-sampling (1:10)"] = eval_sup(hgb(seed, None), sc.inverse_transform(Xs), ys, full_idx)[0]
    imb["Cost-sensitive learning (balanced weights)"] = model_rows[sup_name]
    res["imbalance"] = {k: {kk: vv for kk, vv in v.items() if kk in
                            ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr")}
                        for k, v in imb.items()}

    # ---------------- error analysis (seed-level)
    pred = (R_te >= T2)
    amt = df.iloc[te]["Amount"].to_numpy()
    fnm, tpm = (~pred) & (yte == 1), pred & (yte == 1)
    fpm = pred & (yte == 0)
    res["errors"] = {
        "fn": int(fnm.sum()), "tp": int(tpm.sum()), "fp": int(fpm.sum()),
        "fn_in_monitor_band": int(((R_te >= T1) & fnm).sum()),
        "median_amount_fn": float(np.median(amt[fnm])) if fnm.any() else None,
        "median_amount_tp": float(np.median(amt[tpm])) if tpm.any() else None,
        "median_amount_fp": float(np.median(amt[fpm])) if fpm.any() else None,
        "median_amount_legit": float(np.median(amt[yte == 0])),
        "median_A_fn": float(np.median(A_te[fnm])) if fnm.any() else None,
        "median_A_tp": float(np.median(A_te[tpm])) if tpm.any() else None,
        "median_A_fp": float(np.median(A_te[fpm])) if fpm.any() else None,
        "median_P_fn": float(np.median(P_te[fnm])) if fnm.any() else None,
    }

    # ---------------- permutation importance of HistGB on validation (PR-AUC)
    sub = np.concatenate([np.flatnonzero(yva == 1),
                          rng.choice(np.flatnonzero(yva == 0), min(20000, int((yva == 0).sum())), replace=False)])
    pi = permutation_importance(sup, Xva[sub], yva[sub], scoring="average_precision",
                                n_repeats=5, random_state=seed, n_jobs=1)
    order = np.argsort(-pi.importances_mean)[:10]
    res["perm_importance"] = [[cols_full[i], float(pi.importances_mean[i]),
                               float(pi.importances_std[i])] for i in order]
    return res


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    meta = {"rows_raw": n0, "rows_dedup": len(df), "fraud_dedup": int(df["Class"].sum())}
    print(meta, flush=True)
    allres = {"meta": meta, "seeds": []}
    for s in SEEDS:
        t0 = time.time()
        r = run_seed(df, s, out, extras=True)
        allres["seeds"].append(r)
        print(f"seed {s} done in {time.time() - t0:.0f}s", flush=True)
        for k, v in r["models"].items():
            print(f"  {k:30s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
                  f"ROC={v['roc_auc']:.4f} PR={v['pr_auc']:.3f} FPR={v['fpr']:.5f} "
                  f"lat={v['latency_ms']:.2f}ms tps={v['throughput_tps']:.0f}", flush=True)
        (out / "results.json").write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
