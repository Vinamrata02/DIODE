"""PROB MATLAB wrapper."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

try:
    import matlab
    import matlab.engine
except ImportError:  # pragma: no cover - optional dependency
    matlab = None


def run_prob_from_numpy(
    X_genes_by_cells,
    times,
    prob_dir,
    num_draws=None,
    burn_in=None,
    thin=None,
    average_tied_times=True,
):
    """Run PROB's ODE Bayesian Lasso MATLAB function."""
    if matlab is None:
        raise ImportError("MATLAB Engine for Python is required for PROB.")
    prob_dir = Path(prob_dir)
    X_genes_by_cells = np.asarray(X_genes_by_cells, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if average_tied_times:
        X_genes_by_cells, times = _average_tied_times(X_genes_by_cells, times)
    eng = matlab.engine.start_matlab()
    matlab_stdout = io.StringIO()
    matlab_stderr = io.StringIO()
    try:
        eng.addpath(eng.genpath(str(prob_dir)), nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
        eng.rehash(nargout=0, stdout=matlab_stdout, stderr=matlab_stderr)
        X_mat = matlab.double(X_genes_by_cells.tolist())
        time_mat = matlab.double(times.tolist())
        matlab_args = []
        if num_draws is not None:
            matlab_args.extend(["NumDraws", float(num_draws)])
        if burn_in is not None:
            matlab_args.extend(["BurnIn", float(burn_in)])
        if thin is not None:
            matlab_args.extend(["Thin", float(thin)])
        return np.asarray(
            eng.ODE_BayesianLasso_Python(
                X_mat,
                time_mat,
                *matlab_args,
                stdout=matlab_stdout,
                stderr=matlab_stderr,
            ),
            dtype=float,
        )
    finally:
        eng.quit()


def _average_tied_times(X_genes_by_cells, times):
    X = np.asarray(X_genes_by_cells, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("PROB expects a genes-by-cells expression matrix.")
    if X.shape[1] != times.size:
        raise ValueError("PROB expression columns must match pseudotime length.")
    if not np.isfinite(X).all() or not np.isfinite(times).all():
        raise ValueError("PROB inputs must be finite.")

    order = np.argsort(times, kind="stable")
    X = X[:, order]
    times = times[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if unique_times.size < 2:
        raise ValueError("PROB needs at least two unique pseudotime values.")
    if unique_times.size == times.size:
        return X, times

    averaged = np.zeros((X.shape[0], unique_times.size), dtype=np.float64)
    for idx in range(unique_times.size):
        averaged[:, idx] = X[:, inverse == idx].mean(axis=1)
    return averaged, unique_times
