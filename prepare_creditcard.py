import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT = DATA_DIR / "creditcard.csv"
OUT_ALL = DATA_DIR / "transactions.csv"
OUT_TEST = DATA_DIR / "test.csv"


def main():
    df = pd.read_csv(INPUT)

    # Map columns to our project schema
    df = df.rename(columns={
        "Time": "time_seconds",
        "Amount": "amount",
        "Class": "is_fraud"
    })

    # Create a timestamp from seconds (arbitrary start date)
    start = datetime(2026, 1, 1, 0, 0, 0)
    df["timestamp"] = df["time_seconds"].apply(
        lambda s: (start + timedelta(seconds=float(s))).isoformat()
    )

    # Create synthetic users (ULB dataset doesn't have user_id)
    # This groups transactions into 1000 fake users (change 1000 if you want)
    df["user_id"] = "U" + (df.index % 1000).astype(str)

    # Keep useful columns first
    v_cols = [c for c in df.columns if c.startswith("V")]
    df = df[["timestamp", "user_id", "amount"] + v_cols + ["is_fraud"]]

    # Split into train/test (simple split for now)
    # We'll do 80/20 split
    n = len(df)
    split = int(n * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    train_df.to_csv(OUT_ALL, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    print("Saved:")
    print(" -", OUT_ALL)
    print(" -", OUT_TEST)
    print("\nTrain shape:", train_df.shape, "Test shape:", test_df.shape)
    print("\nSample rows:\n", train_df.head(3))


if __name__ == "__main__":
    main()

