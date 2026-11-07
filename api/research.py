"""
Research results endpoints.

Serves the experiment results produced by the scripts in experiments/ so the frontend can
show the paper's tables and charts. Regenerate with:
    python experiments/build_summary.py results
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.deps import verify_api_key

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

router = APIRouter(prefix="/research", tags=["research"], dependencies=[Depends(verify_api_key)])


def _read(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found. Run the experiments and then "
                   "`python experiments/build_summary.py results`.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/summary")
def research_summary():
    """All aggregated results (model comparison, ablation, imbalance, temporal split,
    anomaly-score features, tuning) for every dataset."""
    return _read(RESULTS_DIR / "summary.json")


@router.get("/summary/{dataset}")
def research_dataset(dataset: str):
    """Aggregated results for one dataset: `creditcard` or `paysim`."""
    summary = _read(RESULTS_DIR / "summary.json")
    if dataset not in summary.get("datasets", {}):
        raise HTTPException(status_code=404, detail=f"No results for dataset '{dataset}'.")
    return summary["datasets"][dataset]
