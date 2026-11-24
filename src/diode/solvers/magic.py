"""MAGIC wrapper for cells-by-genes expression."""

import numpy as np


def run_magic_from_numpy(expression):
    import scanpy as sc
    import scanpy.external as sce

    adata = sc.AnnData(np.asarray(expression, dtype=float).copy())
    sce.pp.magic(adata, name_list="all_genes", random_state=0)
    if hasattr(adata.X, "toarray"):
        return adata.X.toarray()
    return np.asarray(adata.X)
