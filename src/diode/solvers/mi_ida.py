"""MI and IDA R wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from diode.solvers.scode import r_subprocess_env, resolve_rscript


def run_mi_ida_from_numpy(X_genes_by_cells, output_dir, runner_path, *, alpha=0.01, rscript_bin=None):
    """Run the notebook MI/IDA R workflow and load A_MI, A_IDA_min, and A_IDA_mean."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X_genes_by_cells, dtype=float)
    input_path = output_dir / "genes_by_cells.txt"
    np.savetxt(input_path, X, delimiter="	")
    rscript_bin = resolve_rscript(rscript_bin)
    result = subprocess.run(
        [rscript_bin, str(runner_path), str(input_path), str(output_dir), str(X.shape[0]), str(X.shape[1]), str(alpha)],
        capture_output=True,
        text=True,
        check=True,
        env=r_subprocess_env(),
    )
    return {
        "MI": np.loadtxt(output_dir / "A_MI.txt"),
        "IDA_min": np.loadtxt(output_dir / "A_IDA_min.txt"),
        "IDA_mean": np.loadtxt(output_dir / "A_IDA_mean.txt"),
        "result": result,
    }
