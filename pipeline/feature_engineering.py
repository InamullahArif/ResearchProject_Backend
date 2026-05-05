"""
Tabular feature engineering for fraud detection: velocity, rolling stats,
user-behavior deviation, and time-based signals. Categorical encoding is handled
by sklearn ColumnTransformer (OneHotEncoder) in build_preprocessor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CFG


def basic_clean(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    return df.sort_values(time_col)


def add_velocity_features(
    df: pd.DataFrame, user_col: str, time_col: str, amount_col: str
) -> pd.DataFrame:
    """
    Adds:
      - time_since_prev_tx (seconds)
      - rolling_tx_count_1h, rolling_amount_sum_1h
      - rolling_tx_count_24h, rolling_amount_sum_24h
      - amount_zscore_24h (per user, time-based window)
    """
    df = df.copy()
    df = df.sort_values([user_col, time_col])

    df["time_since_prev_tx"] = (
        df.groupby(user_col)[time_col].diff().dt.total_seconds().fillna(-1)
    )

    df.set_index(time_col, inplace=True)

    rolling_1h = (
        df.groupby(user_col)[amount_col]
        .rolling("1h")
        .agg(["count", "sum"])
        .reset_index(level=0, drop=True)
    )
    df["rolling_tx_count_1h"] = rolling_1h["count"].values
    df["rolling_amount_sum_1h"] = rolling_1h["sum"].values

    rolling_24h = (
        df.groupby(user_col)[amount_col]
        .rolling("24h")
        .agg(["count", "sum"])
        .reset_index(level=0, drop=True)
    )
    df["rolling_tx_count_24h"] = rolling_24h["count"].values
    df["rolling_amount_sum_24h"] = rolling_24h["sum"].values

    roll_mean = (
        df.groupby(user_col)[amount_col]
        .rolling("24h")
        .mean()
        .reset_index(level=0, drop=True)
    )
    roll_std = (
        df.groupby(user_col)[amount_col]
        .rolling("24h")
        .std()
        .reset_index(level=0, drop=True)
        .replace(0, np.nan)
    )
    df["amount_zscore_24h"] = ((df[amount_col] - roll_mean) / roll_std).fillna(0.0)

    df.reset_index(inplace=True)
    return df


def add_time_and_behavior_features(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    amount_col: str,
    *,
    rolling_tx_windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Time-based features (hour, day-of-week) and deviation from per-user history:
    rolling mean amount over last N transactions (excluding current),
    expanding mean/std deviation for amount vs user baseline.
    """
    df = df.copy()
    df = df.sort_values([user_col, time_col])

    ts = pd.to_datetime(df[time_col])
    df["hour"] = ts.dt.hour.astype(np.float64)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.float64)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.float64)

    g = df.groupby(user_col, sort=False)[amount_col]
    for w in rolling_tx_windows:
        col = f"rolling_amount_mean_{w}tx"
        df[col] = g.transform(
            lambda x, n=w: x.shift(1).rolling(n, min_periods=1).mean()
        ).fillna(0.0)

    exp_mean = g.transform(lambda x: x.expanding().mean().shift(1))
    exp_std = g.transform(lambda x: x.expanding().std().shift(1)).replace(0, np.nan)
    df["amount_deviation_user_mean"] = (
        (df[amount_col] - exp_mean) / exp_std.replace(0, np.nan)
    ).fillna(0.0)

    return df


def engineer_features(
    df: pd.DataFrame,
    time_col: str,
    user_col: str,
    amount_col: str,
    *,
    use_velocity: bool = True,
) -> pd.DataFrame:
    """Full preprocessing chain used in training and API batch paths."""
    df = basic_clean(df, time_col)
    if use_velocity:
        df = add_velocity_features(df, user_col, time_col, amount_col)
    else:
        df = df.copy()
        df["time_since_prev_tx"] = 0.0
        df["rolling_tx_count_1h"] = 1.0
        df["rolling_amount_sum_1h"] = df[amount_col]
        df["rolling_tx_count_24h"] = 1.0
        df["rolling_amount_sum_24h"] = df[amount_col]
        df["amount_zscore_24h"] = 0.0
    df = add_time_and_behavior_features(df, user_col, time_col, amount_col)
    return df


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    scale_numeric: bool | None = None,
) -> ColumnTransformer:
    """
    One-hot encodes categoricals; numeric columns are imputed then optionally scaled.
    """
    if scale_numeric is None:
        scale_numeric = CFG.SCALE_NUMERIC

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
    transformers: list = []
    if categorical_cols:
        transformers.append(("cat", ohe, categorical_cols))
    if numeric_cols:
        if scale_numeric:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scale", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, numeric_cols))
        else:
            transformers.append(
                ("num", SimpleImputer(strategy="constant", fill_value=0.0), numeric_cols)
            )

    if not transformers:
        raise ValueError("No feature columns for preprocessor (empty numeric and categorical).")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
