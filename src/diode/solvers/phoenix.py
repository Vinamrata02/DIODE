"""PHOENIX adapter for genes-by-cells expression matrices."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_MODEL_PARAMS = {
    "method": "dopri5",
    "neurons_per_layer": 200,
    "optimizer": "adam",
    "batch_size": 100,
    "init_lr": 1e-4,
    "weight_decay": 0.0,
    "val_split": 0.20,
    "noise": 0.0,
    "epochs": 100,
    "normalize_data": False,
    "scale_expression": 1.0,
}


def run_phoenix_from_numpy(
    X_sim,
    t_span,
    out_dir,
    *,
    model_params=None,
    phoenix_root=None,
    device="auto",
    seed=0,
    time_policy="rank_if_tied",
):
    """Train no-prior PHOENIX and return ``(A, CompletedProcess)``.

    ``X_sim`` has genes in rows and cells in columns. Cells are stably sorted
    by pseudotime. As in the five reproduction notebooks, tied times are
    replaced by evenly spaced rank time spanning the original range (or
    integer ranks if all times are equal). Set ``time_policy='error'`` to
    reject ties instead. The returned signed matrix has target genes in rows
    and source/regulator genes in columns.

    Parameters in ``model_params`` override the notebook defaults above.
    Training uses adjacent-cell transition MSE without a reference/prior
    network. The checkpoint with lowest epoch training MSE supplies the scores;
    ``val_split`` reserves transitions but does not select the checkpoint.
    Execution in a separate Python process isolates upstream module names and
    random seeds from other solvers. Checkpoints, history, inputs, and the
    raw/used time axes are saved in ``out_dir``.
    """
    X = np.asarray(X_sim, dtype=float)
    times = np.asarray(t_span, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 2:
        raise ValueError("PHOENIX expects a genes-by-cells matrix with at least two cells.")
    if times.size != X.shape[1]:
        raise ValueError("t_span length must equal the number of cells in X_sim.")
    if not np.isfinite(X).all() or not np.isfinite(times).all():
        raise ValueError("PHOENIX expression and pseudotime must be finite.")
    if time_policy not in {"rank_if_tied", "error"}:
        raise ValueError("time_policy must be 'rank_if_tied' or 'error'.")

    params = dict(DEFAULT_MODEL_PARAMS)
    overrides = dict(model_params or {})
    unknown = overrides.keys() - params.keys()
    if unknown:
        raise ValueError(f"Unknown PHOENIX model_params: {', '.join(sorted(unknown))}")
    params.update(overrides)
    for key in ("epochs", "batch_size", "neurons_per_layer"):
        value = params[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"PHOENIX {key} must be a positive integer.")
        params[key] = int(value)
    if params["optimizer"] not in {"adam", "rmsprop", "sgd", "adagrad"}:
        raise ValueError("PHOENIX optimizer must be adam, rmsprop, sgd, or adagrad.")
    if not 0 <= params["val_split"] < 1:
        raise ValueError("PHOENIX val_split must be in [0, 1).")
    for key in ("init_lr", "scale_expression"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"PHOENIX {key} must be finite and positive.")
    for key in ("noise", "weight_decay"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"PHOENIX {key} must be finite and nonnegative.")
    if not isinstance(params["normalize_data"], bool):
        raise ValueError("PHOENIX normalize_data must be a boolean.")

    if phoenix_root is None:
        phoenix_root = Path(__file__).resolve().parents[3] / "external" / "PHOENIX"
    code_dir = Path(phoenix_root).expanduser().resolve() / "ode_net" / "code"
    required = ("odenet.py", "datahandler.py", "csvreader.py", "figure_saver.py",
                "read_config.py", "torchdiffeq/__init__.py")
    missing = [name for name in required if not (code_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"PHOENIX runtime missing at {code_dir}: {', '.join(missing)}. "
            "Run scripts/setup_external_methods.sh or set phoenix_root."
        )

    order = np.argsort(times, kind="stable")
    X, raw_times = X[:, order], times[order]
    used_times = raw_times.copy()
    if np.any(np.diff(used_times) <= 0):
        if time_policy == "error":
            raise ValueError("PHOENIX requires strictly increasing pseudotime; tied times found.")
        lo, hi = float(raw_times.min()), float(raw_times.max())
        used_times = np.linspace(lo, hi, times.size) if hi > lo else np.arange(times.size, dtype=float)
    # Upstream CSV reader creates float32 times. Remove only an absolute offset
    # to avoid losing short intervals when pseudotime has a large origin.
    integration_times = (used_times - used_times[0]).astype(np.float32)
    if np.any(np.diff(integration_times) <= 0) or not np.isfinite(integration_times).all():
        raise ValueError("PHOENIX time intervals cannot be represented in float32.")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "input.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([X.shape[0], 1])
        writer.writerows(X)
        writer.writerow(integration_times)
    np.savetxt(out_dir / "raw_pseudotime.csv", raw_times, delimiter=",")
    np.savetxt(out_dir / "training_time.csv", used_times, delimiter=",")
    np.savetxt(out_dir / "integration_time.csv", integration_times, delimiter=",")
    np.savetxt(out_dir / "cell_order.csv", order, delimiter=",", fmt="%d")
    settings_path = out_dir / "run_settings.json"
    settings_path.write_text(json.dumps({
        "model_params": params, "device": str(device), "seed": int(seed),
        "time_policy": time_policy, "time_origin": float(used_times[0]),
    }, indent=2))

    worker = Path(__file__).with_name("_phoenix_worker.py")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(worker), str(code_dir), str(out_dir)],
        capture_output=True, text=True, env=env,
    )
    (out_dir / "stdout.log").write_text(result.stdout)
    (out_dir / "stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"PHOENIX failed:\n{result.stdout}\n{result.stderr}")
    A = np.loadtxt(out_dir / "A.csv", delimiter=",", ndmin=2)
    if A.shape != (X.shape[0], X.shape[0]) or not np.isfinite(A).all():
        raise RuntimeError("PHOENIX returned an invalid score matrix.")
    return A, result
