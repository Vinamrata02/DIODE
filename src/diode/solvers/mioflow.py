"""MIOFlow wrapper using the notebook's PCA and pseudotime bin assignment."""

import numpy as np

from diode.solvers._imputation_worker import pseudotime_bin_rows, run_worker


def run_mioflow_from_numpy(expression, pseudotime, source_root):
    return run_worker("MIOFlow", expression, pseudotime, source_root, "mioflow/mioflow.py")


def _fit_mioflow(expression, pseudotime):
    import anndata as ad
    import pandas as pd
    from mioflow import MIOFlow
    from sklearn.decomposition import PCA

    rows = pseudotime_bin_rows(pseudotime)
    counts = [len(indices) for indices in rows]
    labels = np.empty(len(pseudotime), dtype=int)
    for bin_index, indices in enumerate(rows):
        labels[indices] = bin_index

    n_components = min(20, expression.shape[1], expression.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=0)
    embedding = pca.fit_transform(expression).astype(np.float32)
    adata = ad.AnnData(expression.astype(np.float32))
    adata.obs["time_bin"] = pd.Categorical(labels, ordered=True)
    adata.obsm["X_pca"] = embedding
    adata.varm["PCs"] = pca.components_.T.astype(np.float32)
    model = MIOFlow(
        adata, gaga_model=None, gaga_input_key="X_pca", obs_time_key="time_bin",
        hidden_dim=64, use_cuda=False, n_epochs=20, sample_size=min(64, min(counts)),
        n_trajectories=max(counts), n_bins=len(rows), exp_dir="MIOFlow_Results",
    )
    model.fit()
    trajectories = np.einsum("bnp,pg->bng", model.trajectories, pca.components_) + pca.mean_
    if trajectories.shape[1] < max(counts):
        raise ValueError("MIOFlow generated fewer trajectories than cells in the largest pseudotime bin.")
    reconstructed = np.empty_like(expression)
    for bin_index, indices in enumerate(rows):
        reconstructed[indices] = trajectories[bin_index, :len(indices)]
    return reconstructed
