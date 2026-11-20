"""GRNBoost2 through Arboreto, returning target-by-regulator scores."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


def run_grnboost2_from_numpy(
    expression_cells_by_genes,
    gene_names=None,
    *,
    seed=20,
    n_workers=1,
    threads_per_worker=1,
    regressor_kwargs=None,
    early_stop_window_length=None,
    output_dir=None,
):
    """Infer a network using all genes as candidate regulators.

    Matches SCODE_Data3_human_OtherMethods.ipynb: Arboreto's stochastic
    gradient-boosting defaults, seed 20, and an in-process Dask cluster.
    Rows of the returned matrix are targets; columns are regulators, in
    ``gene_names`` order. Self-edges are zero. ``regressor_kwargs`` can
    override Arboreto defaults, for example for a small smoke test.
    """
    if gene_names is None and isinstance(expression_cells_by_genes, pd.DataFrame):
        gene_names = expression_cells_by_genes.columns.tolist()
    expression = np.asarray(expression_cells_by_genes, dtype=float)
    if expression.ndim != 2 or min(expression.shape) < 2:
        raise ValueError("GRNBoost2 requires a cells-by-genes matrix with at least two cells and genes.")
    if not np.isfinite(expression).all():
        raise ValueError("GRNBoost2 expression values must be finite.")
    if gene_names is None:
        gene_names = [f"G{i + 1}" for i in range(expression.shape[1])]
    gene_names = [str(name) for name in gene_names]
    if len(gene_names) != expression.shape[1] or len(set(gene_names)) != len(gene_names):
        raise ValueError("Provide one unique gene name per expression column.")
    if any(not name.strip() for name in gene_names):
        raise ValueError("Gene names must not be empty.")
    for name, value in (("n_workers", n_workers), ("threads_per_worker", threads_per_worker)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if early_stop_window_length is not None and (
        not isinstance(early_stop_window_length, Integral) or early_stop_window_length < 1
    ):
        raise ValueError("early_stop_window_length must be a positive integer.")

    try:
        from arboreto.core import create_graph, SGBM_KWARGS, EARLY_STOP_WINDOW_LENGTH
        from distributed import Client, LocalCluster
    except ImportError as exc:
        raise ImportError(
            "GRNBoost2 requires Arboreto and Dask Distributed. "
            "Install them in the active environment with: python -m pip install -e '.[grnboost2]'"
        ) from exc

    kwargs = dict(SGBM_KWARGS)
    kwargs.update(regressor_kwargs or {})
    window = EARLY_STOP_WINDOW_LENGTH if early_stop_window_length is None else early_stop_window_length
    with LocalCluster(
        n_workers=int(n_workers),
        threads_per_worker=int(threads_per_worker),
        processes=False,
        dashboard_address=None,
    ) as cluster:
        with Client(cluster) as client:
            # Arboreto 0.1.6 otherwise calls from_delayed([]) for the unused
            # metadata graph, which current Dask rejects. Request both graphs
            # but compute only the links, as in the original notebook.
            links_graph, _ = create_graph(
                expression_matrix=expression,
                gene_names=gene_names,
                tf_names=gene_names,
                regressor_type="GBM",
                regressor_kwargs=kwargs,
                client=client,
                include_meta=True,
                early_stop_window_length=int(window),
                seed=int(seed),
            )
            links = client.compute(links_graph, sync=True)

    links = links.loc[:, ["TF", "target", "importance"]].copy()
    if not set(links["TF"]).issubset(gene_names) or not set(links["target"]).issubset(gene_names):
        raise ValueError("GRNBoost2 returned gene names outside the input gene order.")
    if not np.isfinite(links["importance"].to_numpy(dtype=float)).all():
        raise ValueError("GRNBoost2 returned nonfinite importance scores.")
    links = links.loc[links["TF"] != links["target"]].sort_values("importance", ascending=False)
    scores = (
        links.pivot(index="target", columns="TF", values="importance")
        .reindex(index=gene_names, columns=gene_names)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    np.fill_diagonal(scores, 0.0)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        links.to_csv(output / "GRNBoost2_edges.tsv", sep="\t", index=False)
        np.savetxt(output / "A_GRNBoost2_target_by_regulator.txt", scores, delimiter="\t")
    return scores
