"""
Import this module **first** in entry scripts (before numpy/sklearn/torch).

When PyTorch (LLVM OpenMP) and NumPy/scikit-learn (often Intel OpenMP) are both
loaded, threadpoolctl can warn about incompatible OpenMP runtimes. On macOS this
is usually benign; on Linux it can rarely deadlock. Intel documents allowing
duplicate libiomp load via KMP_DUPLICATE_LIB_OK.

See: https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
