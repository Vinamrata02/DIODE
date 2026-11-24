"""ALRA wrapper for already-normalized expression."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from diode.utils.r_runtime import resolve_repo_r_runtime


def resolve_alra_runtime(runner_path, rscript_bin=None):
    """Use the repo-local ALRA setup when available."""
    repo_root = Path(runner_path).absolute().parent.parent.parent
    return resolve_repo_r_runtime(repo_root, "alra", rscript_bin)


def run_alra_from_numpy(expression, runner_path, *, rscript_bin=None):
    runner_path = Path(runner_path)
    if not runner_path.is_file():
        raise FileNotFoundError(f"ALRA runner not found at {runner_path}.")
    rscript, env = resolve_alra_runtime(runner_path, rscript_bin)
    with TemporaryDirectory(prefix="diode-alra-") as directory:
        input_path = Path(directory) / "input.tsv"
        output_path = Path(directory) / "imputed.tsv"
        np.savetxt(input_path, expression, delimiter="\t")
        result = subprocess.run(
            [rscript, str(runner_path), str(input_path), str(output_path)],
            capture_output=True, text=True, env=env,
        )
        if result.returncode == 3:
            raise ImportError(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(f"ALRA failed: {result.stderr.strip() or result.stdout.strip()}")
        return np.loadtxt(output_path, delimiter="\t", ndmin=2)
