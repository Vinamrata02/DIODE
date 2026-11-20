"""SCODE wrapper used by the benchmarking notebooks."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

def resolve_rscript(rscript_bin=None):
    if rscript_bin is None:
        candidates = [shutil.which("Rscript"), "/usr/local/bin/Rscript", "/usr/bin/Rscript"]
        rscript_bin = next((path for path in candidates if path and os.path.exists(path)), None)
    elif not os.path.isabs(rscript_bin):
        rscript_bin = shutil.which(rscript_bin) or rscript_bin
    if rscript_bin is None or not os.path.exists(rscript_bin):
        raise FileNotFoundError("Could not find Rscript. Set rscript_bin='/full/path/to/Rscript'.")
    return rscript_bin


def r_subprocess_env():
    """Return an environment for R subprocesses without MATLAB runtime paths."""
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH")
    if ld_library_path:
        parts = [
            part for part in ld_library_path.split(os.pathsep)
            if "matlab" not in part.lower()
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def run_scode_from_numpy(
    X_sim,
    t_span,
    out_dir,
    *,
    model_params=None,
    scode_r_path="../SCODE/scode/SCODE.R",
    rscript_bin=None,
    work_dir="../Data/datasim",
):
    X_sim = np.asarray(X_sim, dtype=float)
    t_span = np.asarray(t_span, dtype=float).reshape(-1)
    G, C = X_sim.shape
    if t_span.shape[0] != C:
        raise ValueError(f"t_span length {t_span.shape[0]} must equal number of cells C={C} in X_sim")

    out_dir = Path(out_dir)
    work_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    x_path = work_dir / "X_sim.txt"
    t_path = work_dir / "time_sim.txt"
    np.savetxt(x_path, X_sim, delimiter="\t")
    cell_idx = np.arange(1, C + 1, dtype=float).reshape(-1, 1)
    np.savetxt(t_path, np.hstack([cell_idx, t_span.reshape(-1, 1)]), delimiter="\t")

    params = dict(G=G, C=C, d=4, num_iterations=500)
    if model_params:
        params.update(model_params)

    rscript_bin = resolve_rscript(rscript_bin)
    scode_r_path = Path(scode_r_path)
    if not scode_r_path.exists():
        raise FileNotFoundError(f"SCODE.R not found: {scode_r_path}")

    cmd = [
        rscript_bin,
        str(scode_r_path),
        str(x_path),
        str(t_path),
        str(out_dir),
        str(params["G"]),
        str(params["d"]),
        str(params["C"]),
        str(params["num_iterations"]),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=r_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(
            "SCODE failed:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    a_path = out_dir / "A.txt"
    if not a_path.exists():
        raise FileNotFoundError(f"SCODE did not produce {a_path}")
    return np.loadtxt(a_path, delimiter="\t").T, result


def run_scode_repeated_from_numpy(
    X_sim,
    t_span,
    out_dir,
    *,
    num_trials=4,
    model_params=None,
    scode_r_path="../SCODE/scode/SCODE.R",
    rscript_bin=None,
    work_dir="../Data/datasim",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A_trials = []
    trial_results = []
    for trial_idx in range(1, int(num_trials) + 1):
        trial_dir = out_dir / f"trial_{trial_idx:02d}"
        trial_work_dir = Path(work_dir) / f"trial_{trial_idx:02d}"
        A_trial, result = run_scode_from_numpy(
            X_sim,
            t_span,
            trial_dir,
            model_params=model_params,
            scode_r_path=scode_r_path,
            rscript_bin=rscript_bin,
            work_dir=trial_work_dir,
        )
        A_trials.append(A_trial)
        trial_results.append(result)
    A_mean = np.mean(np.stack(A_trials, axis=0), axis=0)
    np.savetxt(out_dir / "meanA.txt", A_mean, delimiter="\t")
    np.savetxt(out_dir / "A.txt", A_mean, delimiter="\t")
    return A_mean, {"trial_results": trial_results, "A_trials": A_trials}
