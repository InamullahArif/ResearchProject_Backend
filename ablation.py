"""
Feature ablation: train with velocity / deviation features toggled off for comparison.

Usage:
  python ablation.py              # full features
  python ablation.py --no-velocity
"""
from __future__ import annotations

import argparse

import runtime_env  # noqa: F401

from config import CFG
from dataset_loaders import load_dataset
from train import train_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-velocity", action="store_true", help="Disable rolling velocity / z-score features")
    args = parser.parse_args()

    path = str(CFG.TRAIN_CSV)
    df = load_dataset(path, CFG.DATASET_PRESET)
    suffix = "full" if not args.no_velocity else "no_velocity"
    print(f"Ablation mode: {suffix}")
    train_models(df, use_velocity_features=not args.no_velocity)


if __name__ == "__main__":
    main()
