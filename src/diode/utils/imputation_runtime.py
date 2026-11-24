"""Share array transfer and pseudotime binning for imputation subprocesses."""

import os
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def run_imputation_subprocess(module, method, expression, pseudotime, source_root, required_file):
    source_root = Path(source_root).expanduser().resolve()
    if not (source_root / required_file).is_file():
        variable = f"DIODE_{method.upper()}_ROOT"
        raise FileNotFoundError(
            f"{method} checkout is missing at {source_root}. "
            f"Set {variable} to a checkout containing {required_file}."
        )

    with TemporaryDirectory(prefix=f"diode-{method}-") as directory:
        workspace = Path(directory)
        np.savez(workspace / "input.npz", expression=expression, pseudotime=pseudotime)
        env = os.environ.copy()
        source_dir = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(source_dir), env.get("PYTHONPATH")) if path
        )
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(
            [sys.executable, "-m", module, str(source_root)],
            cwd=workspace, env=env, capture_output=True, text=True,
        )
        if result.returncode == 3:
            raise ImportError(result.stderr.strip() or f"{method} dependency unavailable.")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"{method} failed in its worker process:\n{detail}")
        return np.load(workspace / "imputed.npy")


def pseudotime_bin_rows(pseudotime, n_bins=8):
    n_bins = max(2, min(n_bins, len(pseudotime)))
    edges = np.unique(np.quantile(pseudotime, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.array([pseudotime.min(), np.median(pseudotime), pseudotime.max()])
    labels = np.digitize(pseudotime, edges[1:-1])
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def run_imputation_entry_point(fit, source_root):
    """Load arrays from the parent process, fit, and write the output array."""
    try:
        import torch

        sys.path.insert(0, source_root)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        with np.load("input.npz") as data:
            np.save("imputed.npy", fit(data["expression"], data["pseudotime"]))
    except ImportError as exc:
        print(exc, file=sys.stderr)
        sys.exit(3)
