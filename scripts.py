"""
Utility scripts for data preparation.

Example:
  python scripts.py clean-creditcard --max-benign 15000
  python scripts.py sample-transactions --n 100
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from config import CFG
from dataset_loaders import apply_uci_creditcard_style


def clean_creditcard_small(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_benign: int = 15_000,
    random_state: int = 42,
) -> Path:
    """
    Load creditcard.csv, clean it, map to project schema, and shrink by keeping
    every fraud row plus up to ``max_benign`` legitimate transactions (stratified
    by retaining full minority class).

    Cleaning steps:
      - strip column names
      - coerce ``Class`` to int; drop rows that are invalid
      - drop all-NaN columns / rows with any NaN in core columns
      - drop exact duplicate rows
      - map Time/Amount/Class → timestamp/amount/is_fraud (+ synthetic user_id)
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = CFG.DATA_DIR / "creditcard_clean_small.csv"
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if "Class" not in df.columns:
        raise ValueError(f"Expected column 'Class' in {input_path}")

    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)
    df = df[df["Class"].isin((0, 1))]

    core = ["Time", "Amount", "Class"] if "Time" in df.columns else ["Amount", "Class"]
    df = df.dropna(subset=[c for c in core if c in df.columns])
    df = df.drop_duplicates()

    df = apply_uci_creditcard_style(df)

    target = CFG.TARGET_COL
    fraud = df[df[target] == 1]
    benign = df[df[target] == 0]

    if max_benign < len(benign):
        benign = benign.sample(n=max_benign, random_state=random_state)

    out = pd.concat([fraud, benign], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    n_fraud = int((out[target] == 1).sum())
    n_ok = int((out[target] == 0).sum())
    print(
        f"Wrote {output_path} — rows={len(out):,} "
        f"(fraud={n_fraud:,}, benign={n_ok:,}); source={input_path}"
    )
    return output_path


def _sample_balanced_rare_positive(
    df: pd.DataFrame,
    stratify_col: str,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Build an ``n``-row sample with both classes when possible.

    ``train_test_split(..., stratify=...)`` allocates minority rows by rounding
    proportional counts, which often yields **zero** fraud rows when fraud is
    very rare and ``n`` is small (e.g. 100 rows from ~0.2% fraud rate).
    """
    y = pd.to_numeric(df[stratify_col], errors="coerce").fillna(0).astype(int)
    fraud = df.loc[y == 1].reset_index(drop=True)
    benign = df.loc[y == 0].reset_index(drop=True)
    nf, nb = len(fraud), len(benign)
    if nf == 0 or nb == 0:
        warnings.warn(
            f"Column {stratify_col!r} is single-class; using uniform random sample.",
            UserWarning,
            stacklevel=2,
        )
        return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    if n >= len(df):
        return df.reset_index(drop=True)
    if n < 2:
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)

    prop = nf / (nf + nb)
    kf = int(round(n * prop))
    kf = min(nf, max(1, kf))
    if kf > n - 1:
        kf = n - 1
    # ``train.py`` uses stratified split only when each class has >=2 rows in *full* y;
    # include at least 2 frauds here when possible so small CSVs still train cleanly.
    if nf >= 2 and n >= 4:
        kf = max(kf, min(2, nf, n - 1))
    kb = min(nb, n - kf)
    while kf + kb < n and kb < nb:
        kb += 1
    while kf + kb < n and kf < nf:
        kf += 1
    if kf + kb < n:
        warnings.warn(
            f"Only {kf + kb} unique rows available ({nf} fraud, {nb} benign); "
            f"returning that many instead of {n}.",
            UserWarning,
            stacklevel=2,
        )
        n = kf + kb
    fr = fraud.sample(n=kf, random_state=random_state, replace=False)
    bn = benign.sample(n=kb, random_state=random_state + 1, replace=False)
    out = pd.concat([fr, bn], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state + 2).reset_index(drop=True)


def sample_transactions_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    n: int = 100,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> Path:
    """
    Read ``transactions.csv``, apply light cleaning (strip column names, drop
    duplicate rows), then write a CSV with at most ``n`` rows (fewer if the
    file is shorter). Default input: ``data/transactions.csv``; default output:
    ``data/transactions_100.csv`` when ``n`` is 100.

    If ``stratify_col`` is set (e.g. ``is_fraud``), uses **class-aware** sampling
    that keeps at least one fraud and one legitimate row when ``n >= 2`` and
    both classes exist (sklearn-style proportional stratify often yields **0**
    fraud rows when the positive rate is tiny and ``n`` is small).
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    if output_path is None:
        output_path = CFG.DATA_DIR / f"transactions_{n}.csv"
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    if len(df) <= n:
        out = df.reset_index(drop=True)
    elif stratify_col and stratify_col in df.columns:
        out = _sample_balanced_rare_positive(df, stratify_col, n, random_state)
    else:
        out = df.sample(n=n, random_state=random_state)

    out = out.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} — rows={len(out):,} (n={n}); source={input_path}")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data prep utilities for Inam-Project-UK.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "clean-creditcard",
        help="Clean data/creditcard.csv and write a smaller stratified CSV.",
    )
    c.add_argument(
        "--input",
        type=Path,
        default=CFG.DATA_DIR / "creditcard.csv",
        help="Path to raw ULB/Kaggle creditcard.csv",
    )
    c.add_argument(
        "--output",
        type=Path,
        default=CFG.DATA_DIR / "creditcard_clean_small.csv",
        help="Where to save the cleaned, reduced dataset",
    )
    c.add_argument(
        "--max-benign",
        type=int,
        default=15_000,
        help="Maximum number of non-fraud rows to keep (all fraud rows are kept)",
    )
    c.add_argument(
        "--seed",
        type=int,
        default=CFG.RANDOM_STATE,
        help="Random seed for benign subsample and shuffle",
    )

    t = sub.add_parser(
        "sample-transactions",
        help="Take N random rows from data/transactions.csv (default N=100).",
    )
    t.add_argument(
        "--input",
        type=Path,
        default=CFG.DATA_DIR / "transactions.csv",
        help="Path to transactions.csv",
    )
    t.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: data/transactions_<N>.csv)",
    )
    t.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to write",
    )
    t.add_argument(
        "--stratify-col",
        type=str,
        default=None,
        help="Optional column name for stratified sampling (e.g. is_fraud)",
    )
    t.add_argument(
        "--seed",
        type=int,
        default=CFG.RANDOM_STATE,
        help="Random seed",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "clean-creditcard":
        clean_creditcard_small(
            args.input,
            args.output,
            max_benign=args.max_benign,
            random_state=args.seed,
        )
    elif args.command == "sample-transactions":
        sample_transactions_csv(
            args.input,
            args.output,
            n=args.n,
            random_state=args.seed,
            stratify_col=args.stratify_col,
        )


if __name__ == "__main__":
    main()
