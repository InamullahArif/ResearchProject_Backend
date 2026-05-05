"""
Load joblib model artifacts with clear errors when NumPy / toolchain versions differ.

Old `*.joblib` files saved under NumPy 2.x often fail under NumPy 1.x (and vice versa)
with: ValueError ... MT19937 ... is not a known BitGenerator ...
"""
from __future__ import annotations

from pathlib import Path

import joblib


def load_joblib(path: Path | str, *, what: str = "model bundle") -> object:
    path = Path(path)
    try:
        return joblib.load(path)
    except ValueError as e:
        err = str(e)
        if "BitGenerator" in err or "MT19937" in err:
            raise RuntimeError(
                f"Cannot load {what} from:\n  {path.resolve()}\n\n"
                "This usually happens after changing NumPy (for example pinning numpy<2 or "
                "upgrading to NumPy 2). Joblib pickles embed NumPy RNG types that are not "
                "compatible across those versions.\n\n"
                "Fix — delete the trained artifacts and regenerate them with your current stack:\n"
                "  • models/fraud_model.joblib\n"
                "  • models/scoring_bundle.joblib\n"
                "  • models/train_report.joblib\n"
                "Then run:  python train.py\n\n"
                f"Original error: {e}"
            ) from e
        raise
