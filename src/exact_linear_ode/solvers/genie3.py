"""GENIE3 and dynGENIE3 wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_dyn_genie3_from_numpy(expression_cells_by_genes, times, dyn_genie3_dir, ntrees=1000, nthreads=1):
    """Run dynGENIE3 after averaging cells with tied pseudotimes."""
    dyn_genie3_dir = Path(dyn_genie3_dir)
    if str(dyn_genie3_dir) not in sys.path:
        sys.path.insert(0, str(dyn_genie3_dir))
    from dynGENIE3 import dynGENIE3

    df_dyn = pd.DataFrame(np.asarray(expression_cells_by_genes, dtype=float))
    df_dyn["time"] = np.asarray(times, dtype=float).reshape(-1)
    grouped_dyn = df_dyn.groupby("time", sort=True).mean()
    time_points_dyn = [grouped_dyn.index.to_numpy(dtype=float)]
    expression_dyn = [grouped_dyn.to_numpy(dtype=float)]
    vim, *_ = dynGENIE3(expression_dyn, time_points_dyn, ntrees=ntrees, nthreads=nthreads)
    return np.asarray(vim, dtype=float)


def run_genie3_from_numpy(expression_cells_by_genes, genie3_dir, gene_names=None, ntrees=1000, nthreads=1):
    """Run static GENIE3 and return a target-by-regulator matrix."""
    genie3_dir = Path(genie3_dir)
    if str(genie3_dir) not in sys.path:
        sys.path.insert(0, str(genie3_dir))
    from GENIE3 import GENIE3

    expression = np.asarray(expression_cells_by_genes, dtype=float)
    if gene_names is None:
        gene_names = [f"G{i}" for i in range(1, expression.shape[1] + 1)]
    vim = GENIE3(
        expression,
        gene_names=gene_names,
        regulators="all",
        tree_method="RF",
        K="sqrt",
        ntrees=ntrees,
        nthreads=nthreads,
    )
    return np.asarray(vim, dtype=float).T
