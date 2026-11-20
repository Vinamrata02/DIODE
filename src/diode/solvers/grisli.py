"""GRISLI MATLAB wrapper."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:
    import matlab
    import matlab.engine
except ImportError:  # pragma: no cover - optional dependency
    matlab = None


def run_grisli_from_numpy(X, grisli_root, **kwargs):
    """Run GRISLI on a NumPy matrix and return its score matrix."""
    return run_grisili_from_numpy(X, grisli_root, **kwargs)


def run_grisili_from_numpy(X, grisli_root, *, R=100, L_array=None, alpha_min=0.3):
    """Run the Python-facing GRISILI adapter on a spacetime NumPy matrix."""
    if matlab is None:
        raise ImportError("MATLAB Engine for Python is required for GRISLI.")
    grisli_root = Path(grisli_root)
    if not (grisli_root / "GRISILI.m").exists():
        raise FileNotFoundError(f"GRISILI.m not found in GRISLI root: {grisli_root}")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("GRISLI expects a cells-by-(1+genes) matrix with time in the first column.")
    if L_array is None:
        L_array = list(range(10, X.shape[1] + 1, 10))

    eng = matlab.engine.start_matlab()
    matlab_stdout = io.StringIO()
    matlab_stderr = io.StringIO()
    try:
        eng.addpath(eng.genpath(str(grisli_root)), nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
        X_mat = matlab.double(X.tolist())
        rank_matrix = np.asarray(
            eng.GRISILI(
                X_mat,
                float(R),
                matlab.double([list(L_array)]),
                float(alpha_min),
                stdout=matlab_stdout,
                stderr=matlab_stderr,
            ),
            dtype=float,
        )
    finally:
        eng.quit()
    return np.nanmax(rank_matrix) + 1.0 - rank_matrix
