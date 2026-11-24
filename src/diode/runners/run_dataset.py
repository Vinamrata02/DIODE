"""Run AUROC or imputation benchmarks on one configured dataset."""

from __future__ import annotations

import argparse
import os
from functools import partial
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from diode.solvers import diode_lbfgs
from diode.utils.evaluations import (
    minmax_lower_is_better_score,
    ordered_fragmentation_k1,
    paper_pos_score,
    vertical_cosine_scores,
)
from diode.utils.io import (
    diode_config,
    load_config,
    load_ordered_expression_dataset,
    resolve_config_repo_root,
    resolve_path,
)


IMPUTATION_METHODS = {
    "diode": "DIODE",
    "scnode": "scNODE",
    "mioflow": "MIOFlow",
    "magic": "MAGIC",
    "alra": "ALRA",
}
IMPUTATION_COLUMNS = ("dataset", "method", "n_runs", "mean_RMSE", "sample_SD_RMSE")
TRAJECTORY_COLUMNS = (
    "dataset", "method", "pos_score", "trajectory_continuity_score", "vertical_cosine_score",
)
IMPUTATION_DATASET_LABELS = {
    "data1": "Mouse ESC-PrE",
    "data2": "Mouse MEF-iN",
    "data3": "Human ESC-DE",
    "data4_epiblast": "Epiblast",
    "data4_trophectoderm": "Trophectoderm",
}


def validate_task(task, masking):
    if task not in ("auroc", "imputation"):
        raise ValueError(f"Unknown benchmark task {task!r}. Choose from auroc and imputation.")
    if task == "auroc" and masking is not None:
        raise ValueError("A masking fraction requires task='imputation'.")
    if task == "imputation" and masking is not None and (
        not isinstance(masking, Real) or isinstance(masking, bool) or not 0 <= masking < 1
    ):
        raise ValueError("Imputation masking must be in [0, 1). Omit it for trajectory scoring.")


def resolve_imputation_solvers(solvers):
    if solvers is None or (isinstance(solvers, str) and solvers == "all"):
        return list(IMPUTATION_METHODS)
    if isinstance(solvers, str):
        solvers = [solvers]
    selected = []
    for solver in solvers:
        key = str(solver).lower()
        if key not in IMPUTATION_METHODS:
            available = ", ".join(IMPUTATION_METHODS.values())
            raise ValueError(f"{solver} is unavailable for the imputation task. Choose from {available}.")
        selected.append(key)
    return selected


def run_dataset(
    config_path, *, task="auroc", solvers=None, masking=None, progress=True,
    skip_unavailable=True, _run_solver=None, _leave_solver_progress=True,
):
    """Run a dataset task. A plain config-path call returns the DIODE AUROC summary."""
    validate_task(task, masking)
    if task == "imputation":
        solvers = resolve_imputation_solvers(solvers)
        if masking in (None, 0):
            return _run_unmasked_imputation(
                config_path, solvers, progress=progress, skip_unavailable=skip_unavailable,
                leave_progress=_leave_solver_progress,
            )
        return _run_imputation(
            config_path, solvers, masking, progress=progress, skip_unavailable=skip_unavailable,
            leave_progress=_leave_solver_progress,
        )
    if solvers is None:
        return diode_lbfgs._tune_diode(config_path)
    return _run_auroc_solvers(
        config_path, solvers, progress=progress, skip_unavailable=skip_unavailable,
        run_solver=_run_solver, leave_progress=_leave_solver_progress,
    )


def _run_auroc_solvers(config_path, solvers, *, progress, skip_unavailable, run_solver, leave_progress):
    from diode.experiment import DIODEBenchmark, SOLVER_LABELS, SolverUnavailableError

    config = load_config(config_path)
    dataset = config["dataset"]
    benchmark = DIODEBenchmark(repo_root=resolve_config_repo_root(config, config_path))
    solvers = benchmark._resolve_solvers(solvers)
    if run_solver is None:
        run_solver = partial(benchmark._run_solver, config_path=config_path)

    bar = DIODEBenchmark._solver_progress_bar(len(solvers), enabled=progress, leave=leave_progress)
    rows = []
    try:
        for solver in solvers:
            if bar is not None:
                bar.set_description(f"Running: {SOLVER_LABELS[solver]}")
                bar.refresh()
            try:
                rows.append(run_solver(dataset, solver))
            except SolverUnavailableError as exc:
                if not skip_unavailable:
                    raise
                rows.append({
                    "dataset": dataset, "solver": solver,
                    "status": "not_available", "message": str(exc),
                })
            finally:
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return {"rows": rows}


def _imputation_baselines(root, config):
    from diode.solvers.alra import run_alra_from_numpy
    from diode.solvers.magic import run_magic_from_numpy
    from diode.solvers.mioflow import run_mioflow_from_numpy
    from diode.solvers.scnode import run_scnode_from_numpy

    scnode_root = os.environ.get("DIODE_SCNODE_ROOT", root / "external/scNODE")
    mioflow_root = os.environ.get("DIODE_MIOFLOW_ROOT", root / "external/MIOFlow")
    rscript = os.environ.get("DIODE_RSCRIPT") or config.get("solvers", {}).get("scode", {}).get("rscript_bin")
    return {
        "alra": lambda observed, times: run_alra_from_numpy(
            observed, root / "external/ALRA/run_alra.R", rscript_bin=rscript,
        ),
        "magic": lambda observed, times: run_magic_from_numpy(observed),
        "mioflow": lambda observed, times: run_mioflow_from_numpy(observed, times, mioflow_root),
        "scnode": lambda observed, times: run_scnode_from_numpy(observed, times, scnode_root),
    }


def _run_imputation(config_path, solvers, masking, *, progress, skip_unavailable, leave_progress):
    from diode.experiment import DIODEBenchmark, SolverUnavailableError

    config = load_config(config_path)
    root = resolve_config_repo_root(config, config_path)
    dataset = config["dataset"]
    data = load_ordered_expression_dataset(
        resolve_path(config.get("imputation_expression", config["expression"]), root),
        resolve_path(config["pseudotime"], root),
        resolve_path(config["genes"], root),
        dtype=diode_lbfgs._observation_dtype(config),
        sort_kind=None,
    )
    truth, times = data["expression_cells_by_genes"], data["pseudotime"]
    if not np.isfinite(truth).all() or not np.isfinite(times).all():
        raise ValueError(f"Imputation requires finite expression and pseudotime for {dataset}.")
    n_hidden = round(masking * truth.size)
    if not 0 < n_hidden < truth.size:
        raise ValueError(f"Masking fraction must hide at least one entry and leave observations for {dataset}.")

    order = np.argsort(times)
    baselines = _imputation_baselines(root, config)
    scores = {solver: [] for solver in solvers}
    failures = {}
    n_runs = 5
    bar = DIODEBenchmark._solver_progress_bar(
        n_runs * len(scores), enabled=progress, leave=leave_progress,
    )
    try:
        for seed in range(n_runs):
            mask = np.zeros(truth.size, dtype=bool)
            mask[np.random.default_rng(seed).choice(truth.size, size=n_hidden, replace=False)] = True
            mask = mask.reshape(truth.shape)
            observed = truth.copy()
            observed[mask] = 0.0
            for solver in scores:
                if solver in failures:
                    continue
                method = IMPUTATION_METHODS[solver]
                if bar is not None:
                    bar.set_description(f"{dataset}: run {seed + 1}/{n_runs}, {method}")
                try:
                    if solver == "diode":
                        predictions = diode_lbfgs._impute_with_diode(config_path, data, observed, mask, order)
                    else:
                        predictions = baselines[solver](observed, times)
                except (ImportError, FileNotFoundError) as exc:
                    if solver == "diode":
                        raise
                    message = f"{method} unavailable: {exc}"
                    if not skip_unavailable:
                        raise SolverUnavailableError(message) from exc
                    failures[solver] = {"dataset": dataset, "method": method, "message": message}
                    if bar is not None:
                        bar.set_description(f"{dataset}: {method} unavailable")
                        bar.update(n_runs - seed)
                    continue

                predictions = np.asarray(predictions)
                if predictions.shape != truth.shape or not np.isfinite(predictions).all():
                    raise ValueError(f"{method} returned invalid predictions for {dataset}, run {seed + 1}.")
                scores[solver].append(float(np.sqrt(np.mean((predictions[mask] - truth[mask]) ** 2))))
                if bar is not None:
                    bar.update()
        if bar is not None:
            bar.set_description(f"{dataset}: imputation complete")
    finally:
        if bar is not None:
            bar.close()

    rows = [
        {
            "dataset": IMPUTATION_DATASET_LABELS.get(dataset, dataset),
            "method": IMPUTATION_METHODS[solver],
            "n_runs": len(values),
            "mean_RMSE": float(np.mean(values)) if len(values) == n_runs else np.nan,
            "sample_SD_RMSE": float(np.std(values, ddof=1)) if len(values) == n_runs else np.nan,
        }
        for solver, values in scores.items()
    ]
    return {"rows": rows, "failures": list(failures.values())}


def _run_unmasked_imputation(config_path, solvers, *, progress, skip_unavailable, leave_progress):
    from diode.experiment import DIODEBenchmark, SolverUnavailableError
    from diode.utils.slingshot import infer_pseudotime_slingshot, resolve_slingshot_runtime

    config = load_config(config_path)
    root = resolve_config_repo_root(config, config_path)
    dataset = config["dataset"]
    trajectory = config.get("trajectory")
    if trajectory is None:
        raise ValueError(f"{dataset} requires trajectory.expression and trajectory.reference_time for unmasked scoring.")
    if not solvers:
        return {"rows": [], "failures": []}

    def load_expression(path):
        return load_ordered_expression_dataset(
            resolve_path(path, root), resolve_path(config["pseudotime"], root),
            resolve_path(config["genes"], root), dtype=diode_lbfgs._observation_dtype(config),
            sort_kind=None,
        )

    data = load_expression(trajectory["expression"])
    observed, times = data["expression_cells_by_genes"], data["pseudotime"]
    reference = observed
    if "reference_expression" in trajectory:
        reference = load_expression(trajectory["reference_expression"])["expression_cells_by_genes"]
    real_time = np.loadtxt(resolve_path(trajectory["reference_time"], root), ndmin=1)
    if real_time.shape != times.shape or not all(
        np.isfinite(values).all() for values in (observed, reference, times, real_time)
    ):
        raise ValueError(f"{dataset} requires finite expression and one reference time label per cell.")
    order = np.argsort(times, kind="stable")
    start_cell = int(np.argmin(real_time))

    try:
        runtime = resolve_slingshot_runtime(root, os.environ.get("DIODE_RSCRIPT"))
    except (ImportError, FileNotFoundError) as exc:
        raise SolverUnavailableError(f"Fig. 7 POS unavailable: {exc}") from exc

    baselines = _imputation_baselines(root, config)
    rows, failures = [], []
    bar = DIODEBenchmark._solver_progress_bar(len(solvers), enabled=progress, leave=leave_progress)
    try:
        for solver in solvers:
            method = IMPUTATION_METHODS[solver]
            if bar is not None:
                bar.set_description(f"Running: {method}")
            row = {
                "dataset": IMPUTATION_DATASET_LABELS.get(dataset, dataset),
                "method": method,
            }
            try:
                if solver == "diode":
                    smooth = load_expression(config["expression"])
                    fit_order = np.argsort(times, kind=diode_config(config).get("sort_kind", "stable"))
                    predictions = diode_lbfgs._fit_imputation_trajectory(
                        config_path, smooth, smooth["expression_cells_by_genes"], fit_order,
                    )
                else:
                    predictions = baselines[solver](observed, times)
            except (ImportError, FileNotFoundError) as exc:
                if solver == "diode":
                    raise
                message = f"{method} unavailable: {exc}"
                if not skip_unavailable:
                    raise SolverUnavailableError(message) from exc
                failures.append({**row, "message": message})
            else:
                predictions = np.asarray(predictions)
                if predictions.shape != reference.shape or not np.isfinite(predictions).all():
                    raise ValueError(f"{method} returned invalid trajectory predictions for {dataset}.")
                pseudotime, _ = infer_pseudotime_slingshot(
                    predictions, repo_root=root, start_cell_index=start_cell, runtime=runtime,
                )
                tau = kendalltau(pseudotime, real_time).statistic
                if tau < 0:
                    pseudotime = pseudotime.max() - pseudotime
                row.update(
                    pos_score=paper_pos_score(pseudotime, real_time),
                    fragmentation_rate_k1=ordered_fragmentation_k1(predictions, order),
                    vertical_cosine_score=vertical_cosine_scores(predictions, reference)["vertical_cosine_score"],
                )
            rows.append(row)
            if bar is not None:
                bar.update()
    finally:
        if bar is not None:
            bar.close()

    continuity = minmax_lower_is_better_score(
        [row.pop("fragmentation_rate_k1", np.nan) for row in rows],
        bounds=trajectory.get("fragmentation_bounds"),
    )
    for row, score in zip(rows, continuity):
        row["trajectory_continuity_score"] = score
    return {"rows": rows, "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one DIODE dataset.")
    parser.add_argument("config", help="Path to a YAML config.")
    args = parser.parse_args(argv)
    result = run_dataset(args.config)
    print(pd.DataFrame([result["metrics"]]).to_string(index=False))


if __name__ == "__main__":
    main()
