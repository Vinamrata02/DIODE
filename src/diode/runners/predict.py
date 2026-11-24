"""Load fitted AUROC outputs as expression trajectories."""

import numpy as np
import pandas as pd
import torch

from diode.solvers.phoenix import predict_phoenix_trajectory
from diode.utils.evaluations import auroc_from_matrices
from diode.utils.io import (
    diode_config,
    load_config,
    load_ordered_expression_dataset,
    optional_reference_dir,
    resolve_path,
)

PREDICTION_SOLVERS = {"diode": "DIODE", "scode": "SCODE", "phoenix": "PHOENIX"}


def predict_fitted_expression(fit_datasets, genes):
    from diode.experiment import DIODEBenchmark

    genes = [genes] if isinstance(genes, str) else list(dict.fromkeys(genes))
    if not genes:
        raise ValueError("Provide at least one gene for prediction.")
    datasets = [
        (name, path, [solver for solver in PREDICTION_SOLVERS if solver in fitted])
        for name, path, fitted in fit_datasets
    ]
    if not any(solvers for _, _, solvers in datasets):
        raise ValueError("Prediction requires a successful DIODE, SCODE, or PHOENIX fit from .run().")

    data = []
    matched_genes = set()
    for name, path, solvers in datasets:
        if not solvers:
            continue
        config = load_config(path)
        root = DIODEBenchmark._resolve_config_repo_root(config, path)
        observed = load_ordered_expression_dataset(
            resolve_path(config["expression"], root),
            resolve_path(config["pseudotime"], root),
            resolve_path(config["genes"], root),
            reference_dir=optional_reference_dir(config, root) if "diode" in solvers else None,
            sort_kind=diode_config(config).get("sort_kind", "stable"),
        )
        names = pd.Index(observed["gene_names"])
        indices = names.get_indexer(genes)
        dataset_genes = [gene for gene, index in zip(genes, indices) if index != -1]
        if not dataset_genes:
            raise ValueError(f"{name} has no fitted expression for {', '.join(genes)}.")
        indices = indices[indices != -1]
        matched_genes.update(dataset_genes)
        times = observed["pseudotime"]
        smooth = observed["expression_cells_by_genes"]
        output_dir = resolve_path(config["output_dir"], root)
        scores = np.full(len(dataset_genes), np.nan)
        if "diode" in solvers:
            _check_gene_order(output_dir / "final_refit_gene_order.csv", names)
        reference = observed.get("reference_networks")
        if reference is not None:
            network = np.load(output_dir / "final_refit_A.npy")
            truth = reference["unsigned"].to_numpy()
            scores = [
                auroc_from_matrices(np.abs(network[i]), truth[i]) if len(np.unique(truth[i])) > 1 else np.nan
                for i in indices
            ]

        for solver in solvers:
            try:
                predicted = _predict_solver(solver, output_dir, config, root, names, times, smooth, indices)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"{name} {PREDICTION_SOLVERS[solver]} fit is incomplete. {exc}") from exc
            if predicted.shape != (len(times), len(dataset_genes)) or not np.isfinite(predicted).all():
                raise ValueError(f"{name} {PREDICTION_SOLVERS[solver]} produced invalid fitted expression.")
            for column, gene in enumerate(dataset_genes):
                data.append(pd.DataFrame({
                    "dataset": name, "solver": PREDICTION_SOLVERS[solver], "gene": gene,
                    "pseudotime": times, "predicted_expression": predicted[:, column],
                    "observed_smooth": smooth[:, indices[column]], "diode_auroc": scores[column],
                }))
    missing = [gene for gene in genes if gene not in matched_genes]
    if missing:
        raise ValueError(f"No fitted dataset contains {', '.join(missing)}.")
    return pd.concat(data, ignore_index=True)


def _check_gene_order(path, expected):
    genes = pd.read_csv(path, dtype=str)["gene"]
    if not np.array_equal(genes, expected):
        raise ValueError(f"Saved gene order differs from the dataset in {path}.")


def _predict_solver(solver, output_dir, config, root, names, times, smooth, indices):
    if solver == "diode":
        saved_times = np.load(output_dir / "final_refit_t_span.npy").reshape(-1)
        if saved_times.shape != times.shape or not np.allclose(saved_times, times):
            raise ValueError(f"Saved DIODE pseudotime differs from the dataset in {output_dir}.")
        return np.load(output_dir / "final_refit_Y_pred.npy", mmap_mode="r")[0][:, indices]

    solver_dir = output_dir / solver
    _check_gene_order(solver_dir / "scores_gene_order.csv", names)
    if solver == "scode":
        # The benchmark's SCODE wrapper transposes the R output for scoring.
        matrix = np.load(solver_dir / "scores_A.npy").T
        return _predict_scode(matrix, times, smooth[0], indices)

    phoenix = config.get("solvers", {}).get("phoenix", {})
    phoenix_root = resolve_path(phoenix.get("phoenix_root", "external/PHOENIX"), root)
    return predict_phoenix_trajectory(solver_dir, phoenix_root, times, smooth[0])[:, indices]


def _predict_scode(matrix, times, initial_state, indices):
    """Follow the reference plot using the AUROC run's averaged SCODE matrix."""
    matrix = torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float32)
    x0 = torch.as_tensor(initial_state, dtype=torch.float32)
    scale = np.max(np.abs(times))
    t = torch.as_tensor(times / scale if scale else times, dtype=torch.float32)
    with torch.no_grad():
        return np.concatenate([
            (torch.matrix_exp(matrix * step[:, None, None])[:, indices] @ x0).numpy()
            for step in t.split(32)
        ])
