"""Infer Fig. 7 pseudotime using the reference notebook's Slingshot workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from diode.utils.r_runtime import resolve_repo_r_runtime


def resolve_slingshot_runtime(repo_root, rscript_bin=None):
    root = Path(repo_root).resolve()
    rscript, env = resolve_repo_r_runtime(root, "slingshot", rscript_bin)
    check = subprocess.run(
        [rscript, "-e",
         'stopifnot(requireNamespace("slingshot", quietly=TRUE), '
         'requireNamespace("DelayedMatrixStats", quietly=TRUE))'],
        cwd=root, capture_output=True, text=True, env=env,
    )
    if check.returncode:
        raise ImportError(
            "Fig. 7 requires the R packages slingshot and DelayedMatrixStats. Run scripts/setup_external_methods.sh. "
            + check.stderr.strip()
        )
    return rscript, env


def _sanitize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected cells x genes expression data.")
    if not np.isfinite(x).all():
        bad = ~np.isfinite(x)
        means = np.nanmean(np.where(bad, np.nan, x), axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        x = x.copy()
        x[bad] = means[np.where(bad)[1]]
    keep = np.var(x, axis=0) > 1e-12
    if keep.sum() >= 2:
        x = x[:, keep]
    if x.shape[1] == 1:
        x = np.column_stack([x[:, 0], x[:, 0]])
    return x


def _row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return x / norms


def _normalize_pseudotime(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Slingshot returned non-finite pseudotime values.")
    values = values - values.min()
    maximum = values.max()
    return values / maximum if maximum > 0 else np.zeros_like(values)


def _cluster_candidates(n_cells: int, requested: int | None = None) -> list[int | None]:
    if requested is not None:
        return [requested]
    candidates = [None, 8, 6, 5, 4, 3, 2]
    result: list[int | None] = []
    for value in candidates:
        if value is None:
            normalized = None
        else:
            normalized = max(2, min(int(value), n_cells - 1))
        if normalized not in result:
            result.append(normalized)
    return result


def infer_pseudotime_slingshot(
    matrix, *, repo_root, start_cell_index, runtime, n_clusters=None,
    row_normalize=False, random_state=1,
):
    """Return cell-aligned, unit-interval pseudotime and the selected fit details."""
    x = _sanitize(matrix)
    if row_normalize:
        x = _row_normalize(x)
    cells = [f"cell_{i + 1:04d}" for i in range(x.shape[0])]
    genes = [f"gene_{j + 1:04d}" for j in range(x.shape[1])]
    start_cell = int(start_cell_index) + 1
    rscript, env = runtime
    script_path = Path(__file__).with_name("run_slingshot.R")
    if not script_path.is_file():
        raise FileNotFoundError(f"Slingshot runner not found at {script_path}.")

    with TemporaryDirectory(prefix="diode-slingshot-") as work:
        expression = Path(work) / "expression.csv"
        output = Path(work) / "pseudotime.csv"
        pd.DataFrame(x.T, index=genes, columns=cells).to_csv(expression)
        for k in _cluster_candidates(len(cells), n_clusters):
            for distance in ("slingshot", "simple"):
                output.unlink(missing_ok=True)
                result = subprocess.run(
                    [rscript, str(script_path), str(expression), str(output),
                     "" if k is None else str(k), str(start_cell), str(random_state), distance],
                    cwd=Path(repo_root).resolve(), capture_output=True, text=True, env=env,
                )
                if result.returncode == 0 and output.is_file():
                    rows = pd.read_csv(output).set_index("cell_id").loc[cells]
                    values = _normalize_pseudotime(rows["pseudotime"].to_numpy(float))
                    n_used = int(rows["n_clusters"].iloc[0]) if "n_clusters" in rows else k
                    return values, f"clusters={n_used}, start_cell={start_cell}, dist_method={distance}"
                error = (result.stderr or result.stdout or "Slingshot produced no output").strip()
    raise RuntimeError(f"Slingshot failed: {error[-2000:]}")
