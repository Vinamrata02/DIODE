"""SWING wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def run_swing_from_numpy(
    expression_cells_by_genes,
    times,
    swing_dir,
    output_path,
    min_lag=1,
    max_lag=3,
    window_width=10,
    n_trees=100,
    n_jobs=-1,
):
    """Run SWING and return a target-by-regulator score matrix."""
    swing_dir = Path(swing_dir)
    if str(swing_dir) not in sys.path:
        sys.path.insert(0, str(swing_dir))
    from Swing import Swing

    expression = np.asarray(expression_cells_by_genes, dtype=float)
    times = np.asarray(times, dtype=float).reshape(-1)
    swing_df = pd.DataFrame(expression, columns=[f"G{i}" for i in range(1, expression.shape[1] + 1)])
    swing_df.insert(0, "Time", times)
    swing_df = swing_df.groupby("Time", as_index=False, sort=True).mean()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    swing_df.to_csv(output_path, sep="	", index=False)

    sg = Swing(
        str(output_path),
        gene_start=1,
        gene_end=None,
        time_label="Time",
        separator="	",
        min_lag=min_lag,
        max_lag=max_lag,
        window_width=window_width,
        window_type="RandomForest",
    )
    sg.create_windows()
    sg.optimize_params()
    sg.fit_windows(n_trees=n_trees, show_progress=False, n_jobs=n_jobs)
    sg.compile_edges()
    sg.make_static_edge_dict(self_edges=False, lag_method="mean_mean")
    ranked_edges = sg.make_sort_df(sg.edge_dict)
    ranked_edges[["Source", "Target"]] = ranked_edges["regulator-target"].apply(pd.Series)

    graph = nx.DiGraph()
    for _, row in ranked_edges.iterrows():
        graph.add_edge(row["Source"], row["Target"], weight=row["mean_importance"])
    node_order = swing_df.columns.tolist()[1:]
    return nx.to_numpy_array(graph, nodelist=node_order, dtype=float).T
