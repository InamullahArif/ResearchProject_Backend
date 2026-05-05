"""
Generate a small synthetic CSV with fraud bursts and amount spikes (stress-test patterns).

Output: data/stress_synthetic.csv (schema compatible with prepare_creditcard-style pipeline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CFG


def main():
    rng = np.random.default_rng(42)
    n = 8000
    base = pd.Timestamp("2026-01-01")
    t = base + pd.to_timedelta(np.sort(rng.integers(0, 3600 * 200, size=n)), unit="s")
    user = np.array([f"U{i % 120}" for i in range(n)])
    amount = rng.lognormal(3.0, 0.6, size=n)
    is_fraud = np.zeros(n, dtype=int)

    # Burst fraud: same user many txs in short window
    burst_users = [f"U{i}" for i in range(5)]
    for bu in burst_users:
        mask = user == bu
        idx = np.where(mask)[0][:15]
        is_fraud[idx] = 1
        amount[idx] = rng.uniform(800, 4000, size=len(idx))

    # Random rare fraud
    rare = rng.choice(np.arange(n), size=40, replace=False)
    is_fraud[rare] = 1
    amount[rare] *= rng.uniform(3, 8, size=len(rare))

    df = pd.DataFrame(
        {
            "timestamp": t,
            "user_id": user,
            "amount": amount,
            "is_fraud": is_fraud,
            "merchant": rng.choice(["A", "B", "C", "D"], size=n),
            "category": rng.choice(["grocery", "transfer", "fuel"], size=n),
            "location": rng.choice(["UK", "US", "PK"], size=n),
        }
    )
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)

    out = CFG.DATA_DIR / "stress_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(df)}  fraud_rate={is_fraud.mean():.4f}")


if __name__ == "__main__":
    main()
