"""In-memory rolling history for real-time single-transaction scoring (demo / single worker)."""
from __future__ import annotations

from collections import deque
from typing import Any

_MAX = 50_000

_history: deque[dict[str, Any]] = deque(maxlen=_MAX)


def append(row: dict[str, Any]) -> None:
    _history.append(row)


def as_dataframe():
    import pandas as pd

    return pd.DataFrame(list(_history))


def clear() -> None:
    _history.clear()
