"""
Load and normalise CSVs for different public fraud benchmarks.

Presets map common column names onto the project schema:
  timestamp, user_id, amount, is_fraud (+ optional V* / merchant / …)
"""
from __future__ import annotations

import pandas as pd

from config import CFG


def load_raw_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_paysim_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    PaySim-style mobile money CSV (typical Kaggle columns).
    Adjust names if your file differs.
    """
    colmap = {
        "step": "time_step",
        "amount": "amount",
        "isFraud": "is_fraud",
        "isFlaggedFraud": "is_flagged",
        "nameOrig": "user_id",
        "nameDest": "counterparty_id",
        "type": "tx_type",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "is_fraud" not in df.columns:
        raise ValueError("PaySim CSV must contain isFraud or is_fraud.")
    # Synthetic timestamp from step (hour index) for rolling features
    if "time_step" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time_step"], unit="h", origin="unix")
    elif "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="min")
    if "user_id" not in df.columns:
        df["user_id"] = "U0"
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df


def apply_uci_creditcard_style(df: pd.DataFrame) -> pd.DataFrame:
    """Raw UCI/Kaggle credit card file: Time, Amount, Class."""
    if "Class" in df.columns and CFG.TARGET_COL not in df.columns:
        df = df.rename(columns={"Class": CFG.TARGET_COL})
    if "Amount" in df.columns and CFG.AMOUNT_COL not in df.columns:
        df = df.rename(columns={"Amount": CFG.AMOUNT_COL})
    if "Time" in df.columns and "timestamp" not in df.columns:
        start = pd.Timestamp("2024-01-01")
        df["timestamp"] = start + pd.to_timedelta(df["Time"], unit="s")
    if CFG.USER_COL not in df.columns:
        df[CFG.USER_COL] = "U" + (df.index % 1000).astype(str)
    return df


def load_dataset(csv_path: str, preset: str = "generic") -> pd.DataFrame:
    """
    preset:
      - generic: read as-is (expects prepared project schema)
      - paysim: PaySim column renames + timestamp
      - uci_creditcard: raw ULB-style single-file
    """
    df = load_raw_csv(csv_path)
    preset = (preset or "generic").lower().strip()
    if preset == "paysim":
        return apply_paysim_mapping(df)
    if preset == "uci_creditcard":
        return apply_uci_creditcard_style(df)
    return df


def ieee_cis_loader_note() -> str:
    return (
        "IEEE-CIS uses separate train_transaction.csv / train_identity.csv with many columns. "
        "Merge on TransactionID in a notebook, map to timestamp/user_id/amount/is_fraud, "
        "then save as data/transactions.csv for this pipeline."
    )
