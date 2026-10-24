"""Notebook-facing benchmark interface."""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from exact_linear_ode.utils.evaluations import aupr_from_matrices, auroc_from_matrices, f1_direction_from_matrices
from exact_linear_ode.utils.io import load_config, load_ordered_expression_dataset, resolve_path, save_model_outputs

try:
    from tqdm.auto import tqdm as auto_tqdm

    try:
        from tqdm.notebook import tqdm as notebook_tqdm
    except ImportError:  # pragma: no cover - notebook backend is optional.
        notebook_tqdm = None
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    auto_tqdm = None
    notebook_tqdm = None

    class _NullTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def set_description(self, *args, **kwargs):
            return None

        def refresh(self):
            return None

        def update(self, n=1):
            return None

        def close(self):
            return None

    def auto_tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


SOLVER_ALIASES = {
    "ours": "ours",
    "exact_linear_ode": "ours",
    "exactlinearode": "ours",
    "lbfgs": "ours",
    "ours_lbfgs": "ours",
    "genie3": "genie3",
    "dyn_genie3": "dyn_genie3",
    "dyngenie3": "dyn_genie3",
    "scode": "scode",
    "phoenix": "phoenix",
    "grisli": "grisli",
    "prob": "prob",
    "swing": "swing",
    "lm": "lm",
    "linear_regression": "lm",
    "linear regression": "lm",
    "lasso": "lasso",
    "lasso_regression": "lasso",
    "lasso regression": "lasso",
    "mi": "mi",
    "ida_min": "ida_min",
    "ida_mean": "ida_mean",
}

SOLVER_NAMES = (
    "ours",
    "genie3",
    "dyn_genie3",
    "scode",
    "phoenix",
    "grisli",
    "prob",
    "swing",
    "lm",
    "lasso",
    "mi",
    "ida_min",
    "ida_mean",
)

RESULT_COLUMNS = (
    "dataset",
    "solver",
    "status",
    "message",
    "time_taken_seconds",
    "auroc",
    "auroc_absA",
    "auroc_perturb",
)


class SolverUnavailableError(RuntimeError):
    """Raised when a known solver needs an external dependency or configured path."""


class _ClassOrInstanceProperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner, obj)


def _default_repo_root():
    cwd = Path.cwd().resolve()
    if (cwd / "configs").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def _discover_dataset_configs(repo_root=None):
    root = _default_repo_root() if repo_root is None else Path(repo_root).resolve()
    configs_dir = root / "configs"
    dataset_configs = {}
    for config_path in sorted(configs_dir.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        dataset = str(config.get("dataset") or config_path.stem)
        dataset_configs[dataset] = config_path.relative_to(root)
    return dataset_configs


class ExactLinearODEBenchmark:
    """Run ExactLinearODE experiments with a compact notebook interface."""

    @_ClassOrInstanceProperty
    def datasets(cls, obj=None):
        """Available dataset names discovered from configs/*.yaml."""
        repo_root = obj.repo_root if obj is not None else None
        return tuple(_discover_dataset_configs(repo_root))

    @_ClassOrInstanceProperty
    def solvers(cls, obj=None):
        """Solvers available through this high-level interface."""
        return SOLVER_NAMES

    def __init__(self, dataset=None, repo_root=None):
        self.dataset = dataset
        self.repo_root = self._resolve_repo_root(repo_root)

    def config_path(self, dataset):
        """Return the config path for a named dataset."""
        dataset_configs = _discover_dataset_configs(self.repo_root)
        try:
            relative_path = dataset_configs[dataset]
        except KeyError as exc:
            available = ", ".join(dataset_configs)
            raise ValueError(f"Unknown dataset {dataset!r}. Available datasets: {available}") from exc
        return self.repo_root / relative_path

    def run(self, solvers=None, datasets=None, skip_unavailable=True, progress=True):
        """Run selected solvers and return a tidy metrics table.

        Parameters
        ----------
        solvers:
            Solver names to evaluate. Defaults to all solvers exposed by
            `ExactLinearODEBenchmark.solvers`. External solvers require their
            package/script/output paths to be configured before they can run.
        datasets:
            Dataset names to evaluate. Defaults to the dataset passed at
            construction time, or all datasets when no dataset was provided.
        skip_unavailable:
            If true, known solvers with missing optional dependencies or paths are
            returned as status rows instead of raising an error.
        progress:
            If true, show nested tqdm progress bars over datasets and methods.
        """
        selected_solvers = self._resolve_solvers(solvers)
        selected_datasets = self._resolve_datasets(datasets)

        rows = []
        dataset_iter = self._progress(selected_datasets, enabled=progress, desc="Datasets", leave=True, position=0)
        last_dataset = selected_datasets[-1] if selected_datasets else None
        for dataset in dataset_iter:
            if hasattr(dataset_iter, "set_description"):
                dataset_iter.set_description(f"Dataset: {dataset}")
            solver_bar = self._solver_progress_bar(
                selected_solvers,
                enabled=progress,
                leave=(dataset == last_dataset),
            )
            try:
                for solver in selected_solvers:
                    if solver_bar is not None:
                        solver_bar.set_description(f"Running: {solver}")
                        solver_bar.refresh()
                    try:
                        rows.append(self._run_solver(dataset, solver))
                    except SolverUnavailableError as exc:
                        if not skip_unavailable:
                            raise
                        rows.append({
                            "dataset": dataset,
                            "solver": solver,
                            "status": "not_available",
                            "message": str(exc),
                        })
                    finally:
                        if solver_bar is not None:
                            solver_bar.update(1)
            finally:
                if solver_bar is not None:
                    solver_bar.close()

        results = pd.DataFrame(rows)
        for column in RESULT_COLUMNS:
            if column not in results.columns:
                results[column] = np.nan
        ordered = list(RESULT_COLUMNS)
        return results.loc[:, ordered] if not results.empty else results

    def run_all(self, solvers=None, skip_unavailable=True, progress=True):
        """Run selected solvers on all configured datasets."""
        return self.run(solvers=solvers, datasets=self.datasets, skip_unavailable=skip_unavailable, progress=progress)

    def _run_solver(self, dataset, solver):
        if solver == "ours":
            from exact_linear_ode.runners.run_dataset import run_dataset

            with contextlib.redirect_stdout(io.StringIO()):
                result = run_dataset(self.config_path(dataset))
            row = dict(result["metrics"])
            row["dataset"] = dataset
            row["solver"] = "ours"
            row["status"] = "ok"
            return row
        try:
            return self._run_baseline_solver(dataset, solver)
        except Exception as exc:
            raise SolverUnavailableError(f"{solver} unavailable: {exc}") from exc

    def _run_baseline_solver(self, dataset, solver):
        start_time = time.perf_counter()
        config_path = self.config_path(dataset)
        config = load_config(config_path)
        root = self._resolve_config_repo_root(config, config_path)
        solver_cfg = self._solver_config(config, solver)
        expression_path = solver_cfg.get("expression", config.get("baseline_expression", config["expression"]))
        data = load_ordered_expression_dataset(
            resolve_path(expression_path, root),
            resolve_path(config["pseudotime"], root),
            resolve_path(config["genes"], root),
            reference_dir=resolve_path(config["reference_network_dir"], root),
            dtype=torch.float64,
        )
        score_matrix = self._compute_baseline_scores(solver, solver_cfg, config, root, data)

        reference = data["reference_networks"]
        unsigned = reference["unsigned"].to_numpy(dtype=int)
        signed = reference["signed"].to_numpy(dtype=int) if reference["signed"] is not None else None
        edge_mask = np.ones_like(unsigned, dtype=bool) & ~np.eye(unsigned.shape[0], dtype=bool)

        output_dir = resolve_path(config["output_dir"], root) / solver
        save_model_outputs(output_dir, A=score_matrix, gene_names=data["gene_names"], stem="scores")
        auroc = auroc_from_matrices(np.abs(score_matrix), unsigned, mask=edge_mask)
        time_taken_seconds = time.perf_counter() - start_time

        row = {
            "dataset": dataset,
            "solver": solver,
            "status": "ok",
            "time_taken_seconds": time_taken_seconds,
            "auroc": auroc,
            "auroc_absA": auroc,
            "aupr_absA": aupr_from_matrices(np.abs(score_matrix), unsigned, mask=edge_mask),
            "output_dir": str(output_dir),
        }
        if signed is not None:
            f1, precision, recall = f1_direction_from_matrices(score_matrix, signed)
            row.update({"signed_f1": f1, "signed_precision": precision, "signed_recall": recall})
        return row

    def _compute_baseline_scores(self, solver, solver_cfg, config, root, data):
        expression_cells_by_genes = data["expression_cells_by_genes"]
        expression_genes_by_cells = data["expression_genes_by_cells"]
        times = data["pseudotime"]
        gene_names = data["gene_names"]
        baseline_dir = resolve_path(config["output_dir"], root) / solver

        if solver == "genie3":
            genie3_dir = self._required_path(solver_cfg, "genie3_dir", solver)
            from exact_linear_ode.solvers.genie3 import run_genie3_from_numpy

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return run_genie3_from_numpy(
                    expression_cells_by_genes,
                    resolve_path(genie3_dir, root),
                    gene_names=gene_names,
                    ntrees=int(solver_cfg.get("ntrees", 1000)),
                    nthreads=int(solver_cfg.get("nthreads", 1)),
                )

        if solver == "dyn_genie3":
            dyn_genie3_dir = self._required_path(solver_cfg, "dyn_genie3_dir", solver)
            from exact_linear_ode.solvers.genie3 import run_dyn_genie3_from_numpy

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return run_dyn_genie3_from_numpy(
                    expression_cells_by_genes,
                    times,
                    resolve_path(dyn_genie3_dir, root),
                    ntrees=int(solver_cfg.get("ntrees", 1000)),
                    nthreads=int(solver_cfg.get("nthreads", 1)),
                )

        if solver == "scode":
            from exact_linear_ode.solvers.scode import run_scode_repeated_from_numpy

            scode_r_path = self._required_path(solver_cfg, "scode_r_path", solver)
            return run_scode_repeated_from_numpy(
                expression_genes_by_cells,
                times,
                baseline_dir,
                num_trials=int(solver_cfg.get("num_trials", 4)),
                model_params=solver_cfg.get("model_params"),
                scode_r_path=resolve_path(scode_r_path, root),
                rscript_bin=solver_cfg.get("rscript_bin"),
                work_dir=baseline_dir / "work",
            )[0]

        if solver == "phoenix":
            from exact_linear_ode.solvers.phoenix import run_phoenix_from_numpy

            return run_phoenix_from_numpy(
                expression_genes_by_cells,
                times,
                baseline_dir,
                phoenix_root=resolve_path(solver_cfg.get("phoenix_root", "external/PHOENIX"), root),
                model_params=solver_cfg.get("model_params"),
                device=solver_cfg.get("device", config.get("device", "auto")),
                seed=int(solver_cfg.get("seed", config.get("seed", 0))),
                time_policy=solver_cfg.get("time_policy", "rank_if_tied"),
            )[0]

        if solver == "grisli":
            grisli_root = self._required_path(solver_cfg, "grisli_root", solver)
            from exact_linear_ode.solvers.grisli import run_grisli_from_numpy

            spacetime = np.column_stack((times, expression_cells_by_genes))
            return run_grisli_from_numpy(
                spacetime,
                resolve_path(grisli_root, root),
                R=int(solver_cfg.get("R", 100)),
                L_array=solver_cfg.get("L_array"),
                alpha_min=float(solver_cfg.get("alpha_min", 0.3)),
            )

        if solver == "prob":
            prob_dir = self._required_path(solver_cfg, "prob_dir", solver)
            from exact_linear_ode.solvers.prob import run_prob_from_numpy

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return run_prob_from_numpy(
                    expression_genes_by_cells,
                    times,
                    resolve_path(prob_dir, root),
                    num_draws=solver_cfg.get("num_draws"),
                    burn_in=solver_cfg.get("burn_in"),
                    thin=solver_cfg.get("thin"),
                    average_tied_times=bool(solver_cfg.get("average_tied_times", True)),
                )

        if solver == "swing":
            swing_dir = self._required_path(solver_cfg, "swing_dir", solver)
            from exact_linear_ode.solvers.swing import run_swing_from_numpy

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return run_swing_from_numpy(
                    expression_cells_by_genes,
                    times,
                    resolve_path(swing_dir, root),
                    baseline_dir / "swing_input.tsv",
                    min_lag=int(solver_cfg.get("min_lag", 1)),
                    max_lag=int(solver_cfg.get("max_lag", 3)),
                    window_width=int(solver_cfg.get("window_width", 10)),
                    n_trees=int(solver_cfg.get("n_trees", 100)),
                    n_jobs=int(solver_cfg.get("n_jobs", -1)),
                )

        if solver in {"lm", "lasso"}:
            runner_path = self._required_path(solver_cfg, "runner_path", solver)
            from exact_linear_ode.solvers.regression import run_r_matrix_method

            input_path = baseline_dir / "cells_by_genes.txt"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(input_path, expression_cells_by_genes, delimiter="\t")
            output_path = baseline_dir / f"A_{solver.upper()}.txt"
            matrix, _ = run_r_matrix_method(
                resolve_path(runner_path, root),
                input_path,
                output_path,
                rscript_bin=solver_cfg.get("rscript_bin"),
            )
            return matrix

        if solver in {"mi", "ida_min", "ida_mean"}:
            runner_path = self._required_path(solver_cfg, "runner_path", solver)
            from exact_linear_ode.solvers.mi_ida import run_mi_ida_from_numpy

            matrices = run_mi_ida_from_numpy(
                expression_genes_by_cells,
                baseline_dir,
                resolve_path(runner_path, root),
                alpha=float(solver_cfg.get("alpha", 0.01)),
                rscript_bin=solver_cfg.get("rscript_bin"),
            )
            key = {"mi": "MI", "ida_min": "IDA_min", "ida_mean": "IDA_mean"}[solver]
            return matrices[key]

        raise ValueError(f"Unsupported solver {solver!r}. Available solvers: {', '.join(self.solvers)}")

    def _resolve_datasets(self, datasets):
        if datasets is not None:
            return list(datasets)
        if self.dataset is not None:
            return [self.dataset]
        return list(self.datasets)

    def _resolve_solvers(self, solvers):
        if solvers is None or solvers == "all":
            return list(self.solvers)
        if isinstance(solvers, str):
            solvers = [solvers]
        resolved = []
        for solver in solvers:
            key = str(solver).lower()
            try:
                resolved.append(SOLVER_ALIASES[key])
            except KeyError as exc:
                available = ", ".join(self.solvers)
                raise ValueError(f"Unsupported solver {solver!r}. Available solvers: {available}") from exc
        return resolved

    @staticmethod
    def _solver_config(config, solver):
        solver_configs = config.get("solvers", {}) or {}
        return solver_configs.get(solver, {}) or {}

    @staticmethod
    def _required_path(solver_cfg, key, solver):
        value = solver_cfg.get(key)
        if not value:
            raise SolverUnavailableError(f"{solver} requires a configured {key}.")
        return value

    @staticmethod
    def _progress(iterable, *, enabled, **kwargs):
        if not enabled:
            return iterable
        return _tqdm()(iterable, dynamic_ncols=True, **kwargs)

    @staticmethod
    def _solver_progress_bar(solvers, *, enabled, leave):
        if not enabled:
            return None
        return _tqdm()(
            total=len(solvers),
            desc="Running",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
            leave=leave,
            mininterval=0,
            miniters=1,
            position=1,
        )

    @staticmethod
    def _resolve_config_repo_root(config, config_path):
        config_path = Path(config_path).resolve()
        if config.get("repo_root"):
            return Path(config["repo_root"]).expanduser().resolve()
        if config_path.parent.name == "configs":
            return config_path.parents[1]
        return Path.cwd().resolve()

    @staticmethod
    def _resolve_repo_root(repo_root):
        if repo_root is not None:
            return Path(repo_root).resolve()
        cwd = Path.cwd().resolve()
        if (cwd / "configs").exists():
            return cwd
        return Path(__file__).resolve().parents[2]


def _tqdm():
    if notebook_tqdm is not None and _running_in_notebook():
        return notebook_tqdm
    return auto_tqdm


def _running_in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
