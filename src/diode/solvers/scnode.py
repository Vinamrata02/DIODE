"""scNODE wrapper using the notebook's pseudotime bins and cell assignment."""

import numpy as np

from diode.solvers._imputation_worker import pseudotime_bin_rows, run_worker


def run_scnode_from_numpy(expression, pseudotime, source_root):
    return run_worker("scNODE", expression, pseudotime, source_root, "optim/running.py")


def _fit_scnode(expression, pseudotime):
    import torch
    from optim.running import constructscNODEModel, scNODEPredict, scNODETrainWithPreTrain

    rows = pseudotime_bin_rows(pseudotime)
    counts = [len(indices) for indices in rows]
    train_data = [torch.tensor(expression[indices], dtype=torch.float32) for indices in rows]
    times = torch.arange(len(rows), dtype=torch.float32)
    model = constructscNODEModel(
        n_genes=expression.shape[1], latent_dim=16,
        enc_latent_list=[64], dec_latent_list=[64], drift_latent_size=[64],
        latent_enc_act="relu", latent_dec_act="relu", drift_act="relu", ode_method="euler",
    )
    model, *_ = scNODETrainWithPreTrain(
        train_data=train_data, train_tps=times, latent_ode_model=model,
        latent_coeff=1.0, epochs=10, iters=100, batch_size=min(64, min(counts)),
        lr=1e-3, pretrain_iters=100, pretrain_lr=1e-3, kl_coeff=0.0,
    )
    predictions = scNODEPredict(
        latent_ode_model=model, first_tp_data=train_data[0], tps=times, n_cells=max(counts),
    )
    reconstructed = np.empty_like(expression)
    for bin_index, indices in enumerate(rows):
        reconstructed[indices] = predictions[:len(indices), bin_index]
    return reconstructed
