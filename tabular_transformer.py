"""Backward-compatible entry point; implementation lives in models.tabular_transformer."""
from models.tabular_transformer import (
    TabularTransformer,
    TabularTransformerPredictor,
    fit_tabular_transformer,
)

__all__ = [
    "TabularTransformer",
    "TabularTransformerPredictor",
    "fit_tabular_transformer",
]
