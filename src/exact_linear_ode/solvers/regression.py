"""Linear-model and Lasso R wrappers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from exact_linear_ode.solvers.scode import r_subprocess_env, resolve_rscript


def run_r_matrix_method(r_script, input_path, output_path, *, rscript_bin=None):
    """Run an R script that consumes an input matrix path and writes an output matrix."""
    rscript_bin = resolve_rscript(rscript_bin)
    r_script = Path(r_script)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [rscript_bin, str(r_script), str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        check=True,
        env=r_subprocess_env(),
    )
    if not output_path.exists():
        raise FileNotFoundError(f"R method did not produce {output_path}")
    return np.loadtxt(output_path), result
