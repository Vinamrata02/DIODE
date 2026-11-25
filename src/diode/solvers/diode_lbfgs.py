"""DIODE model, L-BFGS training, and hyperparameter tuning."""

from __future__ import annotations

import contextlib
import random
import time
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from diode.utils.evaluations import (
    aupr_from_matrices,
    auroc_from_matrices,
    dominant_eigenvalue_sensitivity_scores,
    f1_direction_from_matrices,
)
from diode.utils.io import (
    diode_config,
    load_config,
    load_ordered_expression_dataset,
    load_reference_network,
    optional_reference_dir,
    resolve_config_repo_root,
    resolve_path,
    save_model_outputs,
)


class DIODE:
    """Run DIODE from a dataset config."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.result_ = None
        self.output_dir_ = None
        self.metrics_ = None

    def fit(self):
        """Run the configured training workflow and return the fit results."""
        self.result_ = _tune_diode(self.config_path)
        self.output_dir_ = Path(self.result_["output_dir"])
        self.metrics_ = self.result_["metrics"]
        return self.result_

    def impute(self, mask=None):
        """Return imputed expression in the input's genes-by-cells order.

        A Boolean mask selects entries to impute. With no mask, select
        entries whose absolute ODE residual exceeds the tuned Huber threshold.
        """
        config = load_config(self.config_path)
        root = resolve_config_repo_root(config, self.config_path)
        data = load_ordered_expression_dataset(
            resolve_path(config["expression"], root),
            resolve_path(config["pseudotime"], root),
            resolve_path(config["genes"], root),
            dtype=_observation_dtype(config),
            sort_kind=None,
        )
        observed = data["expression_cells_by_genes"]
        times = data["pseudotime"]
        if not np.isfinite(observed).all() or not np.isfinite(times).all():
            raise ValueError("Imputation requires finite expression and pseudotime values.")

        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != observed.T.shape:
                raise ValueError("mask must be a Boolean matrix with the same genes-by-cells shape as expression.")
            mask = mask.T

        order = np.argsort(times, kind=diode_config(config).get("sort_kind", "stable"))
        return _impute_with_diode(self.config_path, data, observed, mask, order).T

    def evaluate(self, A_true=None, mask=None):
        """Evaluate saved perturbation scores using a supplied or configured reference."""
        reference_dir = None
        if A_true is None:
            config = load_config(self.config_path)
            root = resolve_config_repo_root(config, self.config_path)
            reference_dir = optional_reference_dir(config, root)
            if reference_dir is None:
                raise ValueError(
                    "Provide a reference to run evaluation. Add reference_network_dir to the "
                    "config file and place reference_network.csv in that directory."
                )

        if self.output_dir_ is None:
            if self.result_ is None:
                raise RuntimeError("Call fit() before evaluate().")
            self.output_dir_ = Path(self.result_["output_dir"])

        score_path = self.output_dir_ / "perturb_delta_val_loss_scores.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing perturbation score file: {score_path}")

        scores = np.loadtxt(score_path, delimiter=",")
        if reference_dir is not None:
            genes = pd.read_csv(self.output_dir_ / "final_refit_gene_order.csv", dtype=str)["gene"].tolist()
            A_true = load_reference_network(reference_dir, gene_names=genes)["unsigned"].to_numpy()
        reference = (np.asarray(A_true) != 0).astype(int)
        if scores.shape != reference.shape:
            raise ValueError(f"A_true shape {reference.shape} does not match score shape {scores.shape}.")

        if mask is None:
            mask = np.ones_like(reference, dtype=bool)
            np.fill_diagonal(mask, False)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != reference.shape:
                raise ValueError(f"mask shape {mask.shape} does not match A_true shape {reference.shape}.")

        return {
            "auroc_perturb": auroc_from_matrices(scores, reference, mask=mask),
            "score_path": score_path,
        }


def row_huber_loss(A, B, delta=1.0):
    diff = A - B
    abs_diff = np.abs(diff)

    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean(axis=1)


def col_huber_loss(A, B, delta=1.0):
    diff = A - B
    abs_diff = np.abs(diff)

    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean(axis=0)


class LTIExact_Steady(nn.Module):
    def __init__(self, d, mask, A=None, X_star_init=None, scale=0.1, eye_shift=0.1, dtype=torch.float32):
        super().__init__()
        self.d = d
        self.register_buffer('mask', mask.to(dtype=torch.bool))

        device = self.mask.device
        I = torch.eye(d, dtype=dtype, device=device)
        if A is None:
            M = torch.randn(d, d, dtype=dtype, device=device)
            W0 = (M - M.T) * scale - eye_shift * I
        else:
            A_init = torch.as_tensor(A, dtype=dtype, device=device)
            W0 = A_init * scale - eye_shift * I

        self.W = nn.Parameter(W0 * self.mask)

        if X_star_init is None:
            X_star0 = torch.zeros(1, d, dtype=dtype, device=device)
        else:
            X_star0 = torch.as_tensor(X_star_init, dtype=dtype, device=device)

        self.X_star = nn.Parameter(X_star0)

    @property
    def A_eff(self):
        return self.W * self.mask

    def predict_traj(self, X0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        A = self.A_eff
        device, dtype = A.device, A.dtype
        X_star = self.X_star

        X0 = X0.to(device=device, dtype=dtype)
        t_span = t_span.to(device=device, dtype=dtype)

        At = A.unsqueeze(0).unsqueeze(0) * t_span[..., None, None]
        E = torch.matrix_exp(At)

        X0_shift = X0 - X_star
        Y_centered = torch.einsum(
            'bid,btdk->btk',
            X0_shift[:, None, :],
            E.transpose(-1, -2),
        )
        Y_pred = Y_centered + X_star
        return Y_pred

    def forward(self, X0, t_span):
        return self.predict_traj(X0, t_span)


def make_pseudotime_bins(t_values, n_bins=8):
    t_values = np.asarray(t_values, dtype=float).reshape(-1)
    n_bins = max(2, min(int(n_bins), t_values.shape[0]))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(t_values, quantiles)
    edges = np.unique(edges)
    if edges.shape[0] < 3:
        edges = np.array([t_values.min(), np.median(t_values), t_values.max()], dtype=float)
    bin_ids = np.digitize(t_values, edges[1:-1], right=False)
    bin_centers = np.array(
        [t_values[bin_ids == idx].mean() for idx in range(int(bin_ids.max()) + 1)],
        dtype=float,
    )
    return bin_ids, bin_centers


def stratified_pseudotime_train_val_split(
    t_values,
    train_fraction=0.8,
    n_bins=8,
    seed=0,
    force_train_indices=(0, -1),
):
    t_values = np.asarray(t_values, dtype=float).reshape(-1)
    n_obs = t_values.shape[0]
    rng = np.random.default_rng(seed)

    force_train_indices = np.asarray(force_train_indices, dtype=int).reshape(-1)
    force_train_indices = np.where(force_train_indices < 0, n_obs + force_train_indices, force_train_indices)
    force_train_indices = np.unique(force_train_indices)
    if np.any((force_train_indices < 0) | (force_train_indices >= n_obs)):
        raise ValueError('force_train_indices contains out-of-range entries.')

    bin_ids, bin_centers = make_pseudotime_bins(t_values, n_bins=n_bins)
    train_mask = np.zeros(n_obs, dtype=bool)
    val_mask = np.zeros(n_obs, dtype=bool)
    train_mask[force_train_indices] = True

    for bin_id in np.unique(bin_ids):
        bin_idx = np.flatnonzero(bin_ids == bin_id)
        candidates = np.array([idx for idx in bin_idx if idx not in force_train_indices], dtype=int)
        desired_val = int(round((1.0 - train_fraction) * len(bin_idx)))
        max_val = min(len(candidates), len(bin_idx) - 1)
        n_val = min(desired_val, max_val)
        if n_val > 0:
            chosen_val = rng.choice(candidates, size=n_val, replace=False)
        else:
            chosen_val = np.array([], dtype=int)
        val_mask[chosen_val] = True
        train_mask[np.setdiff1d(bin_idx, chosen_val, assume_unique=False)] = True

    train_mask[force_train_indices] = True
    val_mask[force_train_indices] = False

    if np.any(train_mask & val_mask):
        raise RuntimeError('Train/validation masks overlap.')
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError('Train/validation split produced an empty partition.')

    split_df = pd.DataFrame({
        'obs_index': np.arange(n_obs, dtype=int),
        'pseudotime': t_values,
        'bin_id': bin_ids.astype(int),
        'is_forced_train': np.isin(np.arange(n_obs), force_train_indices),
        'split': np.where(train_mask, 'train', 'validation'),
    })
    split_df['bin_count'] = split_df.groupby('bin_id')['obs_index'].transform('size')

    return {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'train_indices': np.flatnonzero(train_mask),
        'val_indices': np.flatnonzero(val_mask),
        'bin_ids': bin_ids.astype(int),
        'bin_centers': bin_centers,
        'forced_train_indices': force_train_indices,
        'split_df': split_df,
    }


def _masked_data_loss(Y_pred, Y_obs, mask, loss_type='mse', huber_delta=0.5):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=Y_pred.device)
    if mask.ndim != 1:
        raise ValueError('mask must be one-dimensional over time/observations.')
    if int(mask.sum().item()) == 0:
        raise ValueError('mask selects zero observations.')

    Y_pred_sel = Y_pred[:, mask, :]
    Y_obs_sel = Y_obs[:, mask, :]

    if loss_type.lower() == 'mse':
        return F.mse_loss(Y_pred_sel, Y_obs_sel)
    if loss_type.lower() == 'huber':
        return F.huber_loss(Y_pred_sel, Y_obs_sel, delta=huber_delta, reduction='mean')
    raise ValueError("loss_type must be 'mse' or 'huber'.")


def _masked_regression_metrics(Y_pred, Y_obs, mask):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=Y_pred.device)
    if mask.ndim != 1:
        raise ValueError('mask must be one-dimensional over time/observations.')
    if int(mask.sum().item()) == 0:
        raise ValueError('mask selects zero observations.')

    diff = Y_pred[:, mask, :] - Y_obs[:, mask, :]
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    return {
        'mse': mse,
        'rmse': torch.sqrt(torch.clamp(mse, min=0.0)),
        'mae': mae,
    }


def _regularization_term(model, l1_lambda=0.0, l2_lambda=0.0):
    W_masked = model.W * model.mask
    reg = torch.zeros((), device=W_masked.device, dtype=W_masked.dtype)
    if l2_lambda > 0.0:
        reg = reg + l2_lambda * W_masked.pow(2).sum()
    if l1_lambda > 0.0:
        reg = reg + l1_lambda * W_masked.abs().sum()
    return reg


def _gather_flat_grad(model):
    flats = []
    for param in model.parameters():
        if param.grad is None:
            flats.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flats.append(param.grad.reshape(-1))
    return torch.cat(flats)


def _evaluate_model(
    model,
    X0,
    t_span,
    Y_obs,
    train_mask,
    val_mask,
    loss_type,
    huber_delta,
    l1_lambda,
    l2_lambda,
    *,
    compute_val=True,
):
    Y_pred = model(X0, t_span)
    train_data_loss = _masked_data_loss(Y_pred, Y_obs, train_mask, loss_type=loss_type, huber_delta=huber_delta)
    train_metrics = _masked_regression_metrics(Y_pred, Y_obs, train_mask)
    reg = _regularization_term(model, l1_lambda=l1_lambda, l2_lambda=l2_lambda)

    result = {
        'Y_pred': Y_pred,
        'train_data_loss': train_data_loss,
        'train_mse': train_metrics['mse'],
        'train_rmse': train_metrics['rmse'],
        'train_mae': train_metrics['mae'],
        'reg': reg,
        'train_total_loss': train_data_loss + reg,
    }

    if compute_val:
        val_data_loss = _masked_data_loss(Y_pred, Y_obs, val_mask, loss_type=loss_type, huber_delta=huber_delta)
        val_metrics = _masked_regression_metrics(Y_pred, Y_obs, val_mask)
        result['val_data_loss'] = val_data_loss
        result['val_mse'] = val_metrics['mse']
        result['val_rmse'] = val_metrics['rmse']
        result['val_mae'] = val_metrics['mae']
        result['val_total_loss'] = val_data_loss + reg
    else:
        result['val_data_loss'] = None
        result['val_mse'] = None
        result['val_rmse'] = None
        result['val_mae'] = None
        result['val_total_loss'] = None

    return result


def edge_delta_val_loss_matrix(
    model,
    X0,
    t_span,
    Y_obs,
    val_mask,
    edge_mask,
    *,
    loss_type='huber',
    huber_delta=0.5,
    print_every=500,
):
    edge_mask = np.asarray(edge_mask, dtype=bool)
    d0, d1 = edge_mask.shape
    if d0 != model.d or d1 != model.d:
        raise ValueError(f'edge_mask shape {edge_mask.shape} does not match model dimension {(model.d, model.d)}.')

    device = next(model.parameters()).device
    X0 = X0.to(device)
    t_span = t_span.to(device)
    Y_obs = Y_obs.to(device)

    scores = np.zeros((model.d, model.d), dtype=float)
    with torch.no_grad():
        original_W = model.W.detach().clone()
        baseline_pred = model(X0, t_span)
        baseline_loss = float(_masked_data_loss(
            baseline_pred, Y_obs, val_mask, loss_type=loss_type, huber_delta=huber_delta
        ).detach().cpu().item())

        try:
            edge_indices = list(zip(*np.where(edge_mask)))
            for edge_num, (i, j) in enumerate(edge_indices, start=1):
                model.W.copy_(original_W)
                model.W[i, j] = 0.0
                perturbed_pred = model(X0, t_span)
                perturbed_loss = float(_masked_data_loss(
                    perturbed_pred, Y_obs, val_mask, loss_type=loss_type, huber_delta=huber_delta
                ).detach().cpu().item())
                scores[i, j] = perturbed_loss - baseline_loss
                if print_every and (edge_num % int(print_every) == 0 or edge_num == len(edge_indices)):
                    print(f'perturbation delta-val-loss edges: {edge_num}/{len(edge_indices)}')
        finally:
            model.W.copy_(original_W)

    return scores, baseline_loss


def train_lti_lbfgs_with_validation(
    model,
    X0,
    t_span,
    Y_obs,
    train_mask,
    val_mask,
    A_true_list,
    A_true_direction_list,
    Mask_auroc,
    *,
    loss_type='huber',
    huber_delta=2.5,
    l1_lambda=1e-4,
    l2_lambda=1e-7,
    lr=0.3,
    epochs=50,
    print_every=1,
    lbfgs_max_iter=20,
    lbfgs_history_size=10,
    lbfgs_line_search_fn='strong_wolfe',
    optimizer_tolerance_grad=1e-7,
    optimizer_tolerance_change=1e-9,
    stop_tolerance_grad=1e-7,
    stop_tolerance_change=1e-9,
    stop_on_loss=False,
    stop_on_parameter=False,
    stop_on_optimizer_step_size=False,
    stop_on_grad=False,
    patience_level=1,
    device=None,
    report_aupr=False,
):
    if device is None:
        device = next(model.parameters()).device

    X0 = X0.to(device)
    t_span = t_span.to(device)
    Y_obs = Y_obs.to(device)
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        line_search_fn=lbfgs_line_search_fn,
        tolerance_grad=optimizer_tolerance_grad,
        tolerance_change=optimizer_tolerance_change,
    )

    history_rows = []
    best_state_dict = None
    best_val_data_loss = float('inf')
    best_epoch = None
    best_A = None
    closure_calls_total = 0
    t_start = time.perf_counter()
    stopped_early = False
    stop_epoch = None
    stop_reasons = []
    patience_level = max(1, int(patience_level))
    patience_counters = {
        'loss': 0,
        'parameter_A': 0,
        'parameter_X_star': 0,
        'parameter': 0,
        'optimizer_step_size': 0,
        'grad': 0,
    }

    state_param = next(iter(model.parameters()))

    with torch.no_grad():
        initial_eval = _evaluate_model(
            model, X0, t_span, Y_obs, train_mask, val_mask,
            loss_type, huber_delta, l1_lambda, l2_lambda,
        )
    print(
        'initial '
        f"train_total={initial_eval['train_total_loss'].item():.8e} "
        f"train_data={initial_eval['train_data_loss'].item():.8e} "
        f"val_data={initial_eval['val_data_loss'].item():.8e}"
    )

    for epoch in range(epochs):
        closure_trace = []

        with torch.no_grad():
            pre_eval = _evaluate_model(
                model, X0, t_span, Y_obs, train_mask, val_mask,
                loss_type, huber_delta, l1_lambda, l2_lambda,
            )
            A_before = model.A_eff.detach().clone()
            X_star_before = model.X_star.detach().clone()
            epoch_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def closure():
            nonlocal closure_calls_total
            closure_calls_total += 1
            opt.zero_grad(set_to_none=True)
            train_eval = _evaluate_model(
                model, X0, t_span, Y_obs, train_mask, val_mask,
                loss_type, huber_delta, l1_lambda, l2_lambda,
                compute_val=False,
            )
            raw_loss = train_eval['train_total_loss']
            loss_is_finite = bool(torch.isfinite(raw_loss).item())
            if loss_is_finite:
                loss = raw_loss
            else:
                zero_link = torch.zeros((), device=device, dtype=raw_loss.dtype)
                for param in model.parameters():
                    zero_link = zero_link + param.sum() * 0.0
                loss = torch.nan_to_num(raw_loss, nan=1e100, posinf=1e100, neginf=1e100) + zero_link
            loss.backward()
            flat_grad = _gather_flat_grad(model)
            grad_is_finite = bool(torch.isfinite(flat_grad).all().item())
            closure_trace.append({
                'train_total_loss': float(raw_loss.detach().cpu().item()),
                'grad_max_abs': float(flat_grad.abs().max().item()) if grad_is_finite else np.inf,
                'grad_l2_norm': float(flat_grad.norm().item()) if grad_is_finite else np.inf,
                'loss_is_finite': loss_is_finite,
                'grad_is_finite': grad_is_finite,
            })
            return loss

        opt.step(closure)

        with torch.no_grad():
            post_eval = _evaluate_model(
                model, X0, t_span, Y_obs, train_mask, val_mask,
                loss_type, huber_delta, l1_lambda, l2_lambda,
            )
            elapsed_sec = time.perf_counter() - t_start
            A_after = model.A_eff.detach().clone()
            X_star_after = model.X_star.detach().clone()
            post_finite = bool(
                torch.isfinite(A_after).all().item()
                and torch.isfinite(X_star_after).all().item()
                and torch.isfinite(post_eval['train_total_loss']).item()
                and torch.isfinite(post_eval['val_total_loss']).item()
            )

            if not post_finite:
                model.load_state_dict(epoch_state_dict)
                opt.zero_grad(set_to_none=True)
                post_eval = _evaluate_model(
                    model, X0, t_span, Y_obs, train_mask, val_mask,
                    loss_type, huber_delta, l1_lambda, l2_lambda,
                )
                A_after = model.A_eff.detach().clone()
                X_star_after = model.X_star.detach().clone()

            A_eff_np = A_after.cpu().numpy()
            if not np.isfinite(A_eff_np).all():
                raise ValueError(f'Rollback failed: non-finite values remain in A_eff at epoch {epoch}.')

        opt.zero_grad(set_to_none=True)
        grad_eval = _evaluate_model(
            model, X0, t_span, Y_obs, train_mask, val_mask,
            loss_type, huber_delta, l1_lambda, l2_lambda,
            compute_val=False,
        )
        grad_eval['train_total_loss'].backward()
        flat_grad_post = _gather_flat_grad(model)
        post_step_train_grad_max_abs = float(flat_grad_post.abs().max().item())
        post_step_train_grad_l2_norm = float(flat_grad_post.norm().item())
        opt.zero_grad(set_to_none=True)

        opt_state = opt.state[state_param]
        state_step_max_abs = np.nan
        if 'd' in opt_state and 't' in opt_state and opt_state['d'] is not None and opt_state['t'] is not None:
            state_step_max_abs = float((opt_state['d'] * opt_state['t']).abs().max().item())
        state_loss_change_abs = np.nan
        if 'prev_loss' in opt_state and opt_state['prev_loss'] is not None:
            state_loss_change_abs = abs(float(post_eval['train_total_loss'].item()) - float(opt_state['prev_loss']))

        metric_row = {
            'epoch': int(epoch),
            'elapsed_sec': float(elapsed_sec),
            'closure_calls_epoch': int(len(closure_trace)),
            'closure_calls_total': int(closure_calls_total),
            'train_total_loss_pre': float(pre_eval['train_total_loss'].item()),
            'train_total_loss_post': float(post_eval['train_total_loss'].item()),
            'train_data_loss_pre': float(pre_eval['train_data_loss'].item()),
            'train_data_loss_post': float(post_eval['train_data_loss'].item()),
            'train_mse_pre': float(pre_eval['train_mse'].item()),
            'train_mse_post': float(post_eval['train_mse'].item()),
            'train_rmse_pre': float(pre_eval['train_rmse'].item()),
            'train_rmse_post': float(post_eval['train_rmse'].item()),
            'train_mae_pre': float(pre_eval['train_mae'].item()),
            'train_mae_post': float(post_eval['train_mae'].item()),
            'val_total_loss_pre': float(pre_eval['val_total_loss'].item()),
            'val_total_loss_post': float(post_eval['val_total_loss'].item()),
            'val_data_loss_pre': float(pre_eval['val_data_loss'].item()),
            'val_data_loss_post': float(post_eval['val_data_loss'].item()),
            'val_mse_pre': float(pre_eval['val_mse'].item()),
            'val_mse_post': float(post_eval['val_mse'].item()),
            'val_rmse_pre': float(pre_eval['val_rmse'].item()),
            'val_rmse_post': float(post_eval['val_rmse'].item()),
            'val_mae_pre': float(pre_eval['val_mae'].item()),
            'val_mae_post': float(post_eval['val_mae'].item()),
            'regularization_post': float(post_eval['reg'].item()),
            'last_closure_train_total_loss': float(closure_trace[-1]['train_total_loss']) if closure_trace else np.nan,
            'last_closure_grad_max_abs': float(closure_trace[-1]['grad_max_abs']) if closure_trace else np.nan,
            'last_closure_grad_l2_norm': float(closure_trace[-1]['grad_l2_norm']) if closure_trace else np.nan,
            'last_closure_loss_is_finite': bool(closure_trace[-1].get('loss_is_finite', True)) if closure_trace else True,
            'last_closure_grad_is_finite': bool(closure_trace[-1].get('grad_is_finite', True)) if closure_trace else True,
            'post_step_was_finite': bool(post_finite),
            'post_step_train_grad_max_abs': float(post_step_train_grad_max_abs),
            'post_step_train_grad_l2_norm': float(post_step_train_grad_l2_norm),
            'lbfgs_state_step_max_abs': float(state_step_max_abs),
            'lbfgs_state_loss_change_abs': float(state_loss_change_abs),
            'delta_A_fro_norm': float((A_after - A_before).norm().item()),
            'delta_A_max_abs': float((A_after - A_before).abs().max().item()),
            'delta_X_star_l2_norm': float((X_star_after - X_star_before).norm().item()),
            'delta_X_star_max_abs': float((X_star_after - X_star_before).abs().max().item()),
            'meets_tolerance_grad': bool((closure_trace[-1]['grad_max_abs'] <= stop_tolerance_grad) if closure_trace else False),
            'meets_tolerance_change_step': bool(state_step_max_abs <= stop_tolerance_change) if np.isfinite(state_step_max_abs) else False,
            'meets_tolerance_change_loss': bool(state_loss_change_abs <= stop_tolerance_change) if np.isfinite(state_loss_change_abs) else False,
            'stop_triggered': False,
            'stop_reason': '',
        }

        metric_row['meets_parameter_change_A'] = bool(metric_row['delta_A_max_abs'] <= stop_tolerance_change)
        metric_row['meets_parameter_change_X_star'] = bool(metric_row['delta_X_star_max_abs'] <= stop_tolerance_change)
        metric_row['meets_parameter_change'] = bool(
            metric_row['meets_parameter_change_A'] and metric_row['meets_parameter_change_X_star']
        )
        metric_row['likely_no_update_due_to_internal_grad_tol'] = bool(
            metric_row['meets_tolerance_grad']
            and metric_row['delta_A_max_abs'] == 0.0
            and metric_row['delta_X_star_max_abs'] == 0.0
        )

        patience_counters['loss'] = patience_counters['loss'] + 1 if metric_row['meets_tolerance_change_loss'] else 0
        patience_counters['parameter_A'] = (
            patience_counters['parameter_A'] + 1 if metric_row['meets_parameter_change_A'] else 0
        )
        patience_counters['parameter_X_star'] = (
            patience_counters['parameter_X_star'] + 1 if metric_row['meets_parameter_change_X_star'] else 0
        )
        patience_counters['parameter'] = min(
            patience_counters['parameter_A'],
            patience_counters['parameter_X_star'],
        )
        patience_counters['optimizer_step_size'] = patience_counters['optimizer_step_size'] + 1 if metric_row['meets_tolerance_change_step'] else 0
        patience_counters['grad'] = patience_counters['grad'] + 1 if metric_row['meets_tolerance_grad'] else 0

        metric_row['patience_level'] = int(patience_level)
        metric_row['patience_count_loss'] = int(patience_counters['loss'])
        metric_row['patience_count_parameter_A'] = int(patience_counters['parameter_A'])
        metric_row['patience_count_parameter_X_star'] = int(patience_counters['parameter_X_star'])
        metric_row['patience_count_parameter'] = int(patience_counters['parameter'])
        metric_row['patience_count_optimizer_step_size'] = int(patience_counters['optimizer_step_size'])
        metric_row['patience_count_grad'] = int(patience_counters['grad'])

        stop_reasons_epoch = []
        if stop_on_loss and patience_counters['loss'] >= patience_level:
            stop_reasons_epoch.append(
                f'training_loss_change <= stop_tolerance_change for {patience_level} consecutive epochs'
            )
        if (
            stop_on_parameter
            and patience_counters['parameter_A'] >= patience_level
            and patience_counters['parameter_X_star'] >= patience_level
        ):
            stop_reasons_epoch.append(
                f'parameter_change (delta_A_max_abs and delta_X_star_max_abs) <= stop_tolerance_change for {patience_level} consecutive epochs'
            )
        if stop_on_optimizer_step_size and patience_counters['optimizer_step_size'] >= patience_level:
            stop_reasons_epoch.append(
                f'optimizer_step_size (max|d*t|) <= stop_tolerance_change for {patience_level} consecutive epochs'
            )
        if stop_on_grad and patience_counters['grad'] >= patience_level:
            stop_reasons_epoch.append(
                f'training_gradient_max_abs <= stop_tolerance_grad for {patience_level} consecutive epochs'
            )
        if not post_finite:
            stop_reasons_epoch.append('nonfinite_lbfgs_step_rolled_back')
        if stop_reasons_epoch:
            metric_row['stop_triggered'] = True
            metric_row['stop_reason'] = '|'.join(stop_reasons_epoch)

        for i, A_true in enumerate(A_true_list):
            abs_A_eff_np = np.abs(A_eff_np)
            metric_row[f'GS{i}_auroc_absA'] = float(auroc_from_matrices(abs_A_eff_np, A_true, mask=Mask_auroc))
            if report_aupr:
                metric_row[f'GS{i}_aupr_absA'] = float(aupr_from_matrices(abs_A_eff_np, A_true, mask=Mask_auroc))

        for i, A_true_direction in enumerate(A_true_direction_list):
            f1, prec, recall = f1_direction_from_matrices(A_eff_np, A_true_direction)
            metric_row[f'GSDir{i}_f1'] = float(f1)
            metric_row[f'GSDir{i}_precision'] = float(prec)
            metric_row[f'GSDir{i}_recall'] = float(recall)

        history_rows.append(metric_row)

        if metric_row['val_data_loss_post'] < best_val_data_loss:
            best_val_data_loss = metric_row['val_data_loss_post']
            best_epoch = int(epoch)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_A = A_eff_np.copy()

        if epoch % print_every == 0 or epoch == 0 or epoch == epochs - 1 or metric_row['stop_triggered']:
            stop_suffix = ''
            if metric_row['stop_triggered']:
                stop_suffix = f" stop={metric_row['stop_reason']}"
            warning_suffix = ''
            if metric_row['likely_no_update_due_to_internal_grad_tol']:
                warning_suffix = ' warning=internal LBFGS gradient tolerance was already met; this epoch may have had no effective parameter update'
            print(
                f"epoch {epoch:04d} "
                f"train={metric_row['train_data_loss_post']:.8e} "
                f"val={metric_row['val_data_loss_post']:.8e} "
                f"grad_max={metric_row['last_closure_grad_max_abs']:.3e} "
                f"step_max={metric_row['lbfgs_state_step_max_abs']:.3e} "
                f"loss_change={metric_row['lbfgs_state_loss_change_abs']:.3e} "
                f"dA={metric_row['delta_A_fro_norm']:.3e} "
                f"dX*={metric_row['delta_X_star_l2_norm']:.3e} "
                f"best_val={best_val_data_loss:.8e}@{best_epoch}"
                f"{stop_suffix}"
                f"{warning_suffix}"
            )

        if metric_row['stop_triggered']:
            stopped_early = True
            stop_epoch = int(epoch)
            stop_reasons = stop_reasons_epoch.copy()
            break

    history_df = pd.DataFrame(history_rows)
    summary = {
        'best_val_data_loss': float(best_val_data_loss),
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'best_state_dict': best_state_dict,
        'best_A': best_A,
        'history_df': history_df,
        'optimizer_tolerance_grad': float(optimizer_tolerance_grad),
        'optimizer_tolerance_change': float(optimizer_tolerance_change),
        'stop_tolerance_grad': float(stop_tolerance_grad),
        'stop_tolerance_change': float(stop_tolerance_change),
        'stopped_early': bool(stopped_early),
        'stop_epoch': int(stop_epoch) if stop_epoch is not None else None,
        'stop_reason': '|'.join(stop_reasons),
    }
    return summary


def build_selection_summary(history_df, stop_tolerance_change):
    rows = []

    def add_row(criterion, row):
        if row is None:
            rows.append({'criterion': criterion, 'epoch': np.nan})
            return
        record = {
            'criterion': criterion,
            'epoch': int(row['epoch']),
            'train_data_loss_post': float(row['train_data_loss_post']),
            'val_data_loss_post': float(row['val_data_loss_post']),
            'train_mse_post': float(row['train_mse_post']),
            'val_mse_post': float(row['val_mse_post']),
            'train_mae_post': float(row['train_mae_post']),
            'val_mae_post': float(row['val_mae_post']),
            'last_closure_grad_max_abs': float(row['last_closure_grad_max_abs']),
            'lbfgs_state_step_max_abs': float(row['lbfgs_state_step_max_abs']),
            'lbfgs_state_loss_change_abs': float(row['lbfgs_state_loss_change_abs']),
            'delta_A_fro_norm': float(row['delta_A_fro_norm']),
            'delta_A_max_abs': float(row['delta_A_max_abs']),
            'delta_X_star_l2_norm': float(row['delta_X_star_l2_norm']),
            'delta_X_star_max_abs': float(row['delta_X_star_max_abs']),
        }
        for col in history_df.columns:
            if col.startswith('GS'):
                record[col] = float(row[col])
        rows.append(record)

    add_row('min_validation_data_loss', history_df.loc[history_df['val_data_loss_post'].idxmin()])
    add_row('min_training_data_loss', history_df.loc[history_df['train_data_loss_post'].idxmin()])
    add_row('min_delta_A_fro_norm', history_df.loc[history_df['delta_A_fro_norm'].idxmin()])
    add_row('min_delta_X_star_l2_norm', history_df.loc[history_df['delta_X_star_l2_norm'].idxmin()])

    tol_grad_df = history_df[history_df['meets_tolerance_grad']]
    add_row('first_meets_tolerance_grad', tol_grad_df.iloc[0] if not tol_grad_df.empty else None)

    tol_step_df = history_df[history_df['meets_tolerance_change_step']]
    add_row('first_meets_tolerance_change_step', tol_step_df.iloc[0] if not tol_step_df.empty else None)

    tol_loss_df = history_df[history_df['meets_tolerance_change_loss']]
    add_row('first_meets_tolerance_change_loss', tol_loss_df.iloc[0] if not tol_loss_df.empty else None)

    delta_A_norm_df = history_df[history_df['delta_A_fro_norm'] <= stop_tolerance_change]
    add_row('first_meets_delta_A_norm', delta_A_norm_df.iloc[0] if not delta_A_norm_df.empty else None)

    delta_A_max_abs_df = history_df[history_df['delta_A_max_abs'] <= stop_tolerance_change]
    add_row('first_meets_delta_A_max_abs', delta_A_max_abs_df.iloc[0] if not delta_A_max_abs_df.empty else None)

    delta_X_star_norm_df = history_df[history_df['delta_X_star_l2_norm'] <= stop_tolerance_change]
    add_row('first_meets_delta_X_star_norm', delta_X_star_norm_df.iloc[0] if not delta_X_star_norm_df.empty else None)

    delta_X_star_max_abs_df = history_df[history_df['delta_X_star_max_abs'] <= stop_tolerance_change]
    add_row('first_meets_delta_X_star_max_abs', delta_X_star_max_abs_df.iloc[0] if not delta_X_star_max_abs_df.empty else None)

    if 'GS0_auroc_absA' in history_df.columns:
        add_row('oracle_max_GS0_auroc', history_df.loc[history_df['GS0_auroc_absA'].idxmax()])
    if 'GS1_auroc_absA' in history_df.columns:
        add_row('oracle_max_GS1_auroc', history_df.loc[history_df['GS1_auroc_absA'].idxmax()])

    return pd.DataFrame(rows)

def stratified_pseudotime_kfold_split(
    t_values,
    n_splits=5,
    n_bins=8,
    seed=0,
    force_train_indices=(),
):
    t_values = np.asarray(t_values, dtype=float).reshape(-1)
    n_obs = t_values.shape[0]
    rng = np.random.default_rng(seed)

    force_train_indices = np.asarray(force_train_indices, dtype=int).reshape(-1)
    if force_train_indices.size:
        force_train_indices = np.where(force_train_indices < 0, n_obs + force_train_indices, force_train_indices)
        force_train_indices = np.unique(force_train_indices)
        if np.any((force_train_indices < 0) | (force_train_indices >= n_obs)):
            raise ValueError('force_train_indices contains out-of-range entries.')

    bin_ids, _ = make_pseudotime_bins(t_values, n_bins=n_bins)
    fold_ids = np.full(n_obs, -1, dtype=int)

    for bin_id in np.unique(bin_ids):
        bin_idx = np.flatnonzero(bin_ids == bin_id)
        candidates = np.array([idx for idx in bin_idx if idx not in force_train_indices], dtype=int)
        if candidates.size == 0:
            continue
        rng.shuffle(candidates)
        for local_i, obs_idx in enumerate(candidates):
            fold_ids[obs_idx] = int(local_i % n_splits)

    remaining = np.setdiff1d(np.arange(n_obs), force_train_indices)
    if np.any(fold_ids[remaining] < 0):
        raise RuntimeError('Some non-forced observations were not assigned to any fold.')

    fold_masks = []
    for fold_id in range(n_splits):
        heldout_mask = fold_ids == fold_id
        train_mask = ~heldout_mask
        train_mask[force_train_indices] = True
        heldout_mask[force_train_indices] = False
        if not heldout_mask.any():
            raise RuntimeError(f'Fold {fold_id} has no heldout observations.')
        fold_masks.append({
            'fold_id': fold_id + 1,
            'train_mask': train_mask.copy(),
            'heldout_mask': heldout_mask.copy(),
            'train_indices': np.flatnonzero(train_mask),
            'heldout_indices': np.flatnonzero(heldout_mask),
        })

    fold_df = pd.DataFrame({
        'local_index': np.arange(n_obs, dtype=int),
        'pseudotime': t_values,
        'bin_id': bin_ids.astype(int),
        'fold_id': np.where(fold_ids >= 0, fold_ids + 1, 0),
        'is_forced_train': np.isin(np.arange(n_obs), force_train_indices),
    })
    fold_df['fold_label'] = np.where(fold_df['is_forced_train'], 'always_train', 'fold_' + fold_df['fold_id'].astype(str))
    return {
        'fold_masks': fold_masks,
        'fold_df': fold_df,
        'forced_train_indices': force_train_indices,
    }


def extract_stop_row_from_history(history_df):
    stop_df = history_df[history_df['stop_triggered']]
    if not stop_df.empty:
        return stop_df.iloc[0].copy()
    return history_df.iloc[-1].copy()


def flatten_masked_abs_edges(A_matrix, edge_mask):
    A_matrix = np.asarray(A_matrix, dtype=float)
    edge_mask = np.asarray(edge_mask, dtype=bool)
    return np.abs(A_matrix)[edge_mask].ravel()


def summarize_rank_stability(edge_vectors):
    n_runs = len(edge_vectors)
    pair_rows = []
    sim = np.full((n_runs, n_runs), np.nan, dtype=float)

    valid_indices = [idx for idx, vec in enumerate(edge_vectors) if vec is not None and np.isfinite(vec).all()]
    for i, j in combinations(valid_indices, 2):
        if np.ptp(edge_vectors[i]) == 0.0 or np.ptp(edge_vectors[j]) == 0.0:
            rho = 0.0
        else:
            rho, _ = spearmanr(edge_vectors[i], edge_vectors[j])
        if not np.isfinite(rho):
            rho = 0.0
        sim[i, j] = rho
        sim[j, i] = rho
        pair_rows.append({'fold_i': i + 1, 'fold_j': j + 1, 'spearman_rho': float(rho)})

    mean_by_fold = np.nanmean(sim, axis=1)
    finite_mean_mask = np.isfinite(mean_by_fold)
    if finite_mean_mask.any():
        representative_idx = int(np.nanargmax(mean_by_fold))
    elif valid_indices:
        representative_idx = int(valid_indices[0])
    else:
        representative_idx = 0

    return {
        'pairwise_df': pd.DataFrame(pair_rows),
        'pairwise_values': np.asarray([row['spearman_rho'] for row in pair_rows], dtype=float),
        'mean_by_fold': mean_by_fold,
        'representative_idx': representative_idx,
        'valid_fold_count': int(len(valid_indices)),
        'invalid_fold_count': int(n_runs - len(valid_indices)),
    }


def format_hparam(value):
    return f'{value:g}' if value != 0 else '0'


def impute_diode(observed, mask, pseudotime, ode_predictions):
    """Fill masked entries by pseudotime interpolation with an ODE fallback."""
    order = np.argsort(pseudotime)
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    imputed = observed.copy()
    for gene in np.flatnonzero(mask.any(axis=0)):
        unmasked_positions = np.flatnonzero(~mask[order, gene])
        for cell in np.flatnonzero(mask[:, gene]):
            split = np.searchsorted(unmasked_positions, rank[cell])
            if split == 0 or split == len(unmasked_positions):
                imputed[cell, gene] = ode_predictions[cell, gene]
                continue
            left, right = order[unmasked_positions[split - 1 : split + 1]]
            elapsed = pseudotime[right] - pseudotime[left]
            if elapsed == 0:
                imputed[cell, gene] = ode_predictions[cell, gene]
                continue
            slope = (observed[right, gene] - observed[left, gene]) / elapsed
            imputed[cell, gene] = observed[left, gene] + slope * (pseudotime[cell] - pseudotime[left])
    return imputed


def _torch_dtype(name):
    if name in (torch.float64, "float64", "double"):
        return torch.float64
    if name in (torch.float32, "float32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def _resolve_device(device_name):
    """Resolve cpu/cuda/auto device settings."""
    device_name = str(device_name).strip().lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested device: cuda, but CUDA is not available.")
    return torch.device(device_name)


def _as_edge_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    return mask & ~np.eye(mask.shape[0], dtype=bool)


def _make_default_train_val_masks(n_obs, val_fraction=0.2, preferred_val_offset=1):
    """Match the notebooks' alternating 80/20 validation split."""
    val_fraction = float(val_fraction)
    val_count_target = int(round(val_fraction * n_obs))
    val_count_target = max(1, min(n_obs - 2, val_count_target))
    period = max(2, int(round(1.0 / val_fraction)))
    preferred_offset = int(preferred_val_offset) % period
    candidate_offsets = [preferred_offset] + [offset for offset in range(period) if offset != preferred_offset]

    best_val_indices = None
    best_offset = None
    best_score = None
    for offset in candidate_offsets:
        candidate = np.arange(offset, n_obs, period, dtype=int)
        candidate = candidate[(candidate != 0) & (candidate != n_obs - 1)]
        score = (abs(candidate.size - val_count_target), int(candidate.size == 0))
        if best_score is None or score < best_score:
            best_score = score
            best_val_indices = candidate
            best_offset = offset

    val_mask = np.zeros(n_obs, dtype=bool)
    val_mask[best_val_indices] = True
    train_mask = ~val_mask
    train_mask[0] = True
    train_mask[-1] = True
    val_mask[0] = False
    val_mask[-1] = False

    split_df = pd.DataFrame({
        "obs_index": np.arange(n_obs, dtype=int),
        "split": np.where(train_mask, "train", "validation"),
        "is_forced_train": np.isin(np.arange(n_obs), [0, n_obs - 1]),
        "split_mode": "alternating_80_20",
        "validation_period": period,
        "validation_offset": best_offset,
    })
    return train_mask, val_mask, split_df


def _prepare_masks(config, n_obs, time_values):
    train_mask = np.asarray(config.get("train_mask", []), dtype=bool)
    val_mask = np.asarray(config.get("val_mask", []), dtype=bool)
    if train_mask.size and val_mask.size:
        split_df = pd.DataFrame({
            "obs_index": np.arange(n_obs, dtype=int),
            "pseudotime": np.asarray(time_values, dtype=float),
            "split": np.where(train_mask, "train", "validation"),
            "is_forced_train": False,
        })
        return train_mask, val_mask, split_df

    split_cfg = config.get("split", {})
    train_mask, val_mask, split_df = _make_default_train_val_masks(
        n_obs,
        val_fraction=float(split_cfg.get("val_fraction", 0.2)),
        preferred_val_offset=int(split_cfg.get("preferred_val_offset", 1)),
    )
    split_df.insert(1, "pseudotime", np.asarray(time_values, dtype=float))
    if not config.get("cv", {}).get("force_endpoints_in_training", True):
        # Keep endpoints in the outer training partition, but allow CV to hold
        # them out, matching the source notebook's empty forced_train_indices.
        split_df["is_forced_train"] = False
    return train_mask, val_mask, split_df


def _build_cv_fold_masks(config, train_mask_global, val_mask_global, time_values, split_df):
    cv_cfg = config.get("cv", {})
    train_indices_global = np.flatnonzero(train_mask_global)
    train_times = np.asarray(time_values, dtype=float)[train_indices_global]
    forced_global = split_df.loc[split_df["is_forced_train"], "obs_index"].to_numpy(dtype=int)
    forced_local = np.flatnonzero(np.isin(train_indices_global, forced_global))

    cv_result = stratified_pseudotime_kfold_split(
        train_times,
        n_splits=int(cv_cfg.get("n_splits", 5)),
        n_bins=int(cv_cfg.get("n_bins", 8)),
        seed=int(cv_cfg.get("seed", 0)),
        force_train_indices=forced_local,
    )

    cv_fold_masks = []
    for payload in cv_result["fold_masks"]:
        fold_train_mask_global = np.zeros_like(train_mask_global, dtype=bool)
        fold_train_mask_global[train_indices_global[payload["train_indices"]]] = True
        cv_fold_masks.append({
            "fold_id": payload["fold_id"],
            "train_mask_global": fold_train_mask_global,
            "heldout_indices_global": train_indices_global[payload["heldout_indices"]],
            "train_indices_global": train_indices_global[payload["train_indices"]],
        })

    cv_fold_df = cv_result["fold_df"].copy()
    cv_fold_df["global_obs_index"] = train_indices_global[cv_fold_df["local_index"].to_numpy(dtype=int)]
    cv_fold_df["outer_split"] = "outer_train"
    cv_fold_df.loc[cv_fold_df["global_obs_index"].isin(np.flatnonzero(val_mask_global)), "outer_split"] = "outer_validation"
    return cv_fold_masks, cv_fold_df


def _train_kwargs(train_cfg, *, huber_delta, l2_lambda, tolerance_change, device):
    return {
        "loss_type": train_cfg.get("loss_type", "huber"),
        "huber_delta": float(huber_delta),
        "l1_lambda": float(train_cfg.get("l1_lambda", 0.0)),
        "l2_lambda": float(l2_lambda),
        "lr": float(train_cfg.get("lr", 1.0)),
        "epochs": int(train_cfg.get("epochs", 1001)),
        "print_every": int(train_cfg.get("print_every", 500)),
        "lbfgs_max_iter": int(train_cfg.get("lbfgs_max_iter", 20)),
        "lbfgs_history_size": int(train_cfg.get("lbfgs_history_size", 10)),
        "lbfgs_line_search_fn": train_cfg.get("lbfgs_line_search_fn", "strong_wolfe"),
        "optimizer_tolerance_grad": float(train_cfg.get("optimizer_tolerance_grad", 1e-3)),
        "optimizer_tolerance_change": float(tolerance_change),
        "stop_tolerance_grad": float(train_cfg.get("stop_tolerance_grad", 1e-3)),
        "stop_tolerance_change": float(tolerance_change),
        "stop_on_loss": bool(train_cfg.get("stop_on_loss", True)),
        "stop_on_parameter": bool(train_cfg.get("stop_on_parameter", True)),
        "stop_on_optimizer_step_size": bool(train_cfg.get("stop_on_optimizer_step_size", False)),
        "stop_on_grad": bool(train_cfg.get("stop_on_grad", False)),
        "patience_level": int(train_cfg.get("patience_level", 5)),
        "device": device,
        "report_aupr": bool(train_cfg.get("report_aupr", False)),
    }


def _new_model(d, mask, x_star_init, dtype, device):
    return LTIExact_Steady(d, mask, A=0, X_star_init=x_star_init, dtype=dtype).to(device)


def _hyperparameter_grid(grid, tolerance_change):
    rows = list(product(grid["huber_delta"], grid["l2_lambda"]))
    df = pd.DataFrame(rows, columns=["huber_delta", "l2_lambda"])
    df["tolerance_change"] = tolerance_change
    df.insert(0, "combo_id", np.arange(1, len(df) + 1, dtype=int))
    df["run_name"] = df.apply(
        lambda row: (
            f"hd_{format_hparam(row['huber_delta'])}"
            f"_l2_{format_hparam(row['l2_lambda'])}"
            f"_tc_{format_hparam(row['tolerance_change'])}"
        ),
        axis=1,
    )
    return df


def _score_matrix(A, unsigned, signed, edge_mask):
    metrics = {
        "auroc_absA": auroc_from_matrices(np.abs(A), unsigned, mask=edge_mask),
        "aupr_absA": aupr_from_matrices(np.abs(A), unsigned, mask=edge_mask),
    }
    if signed is not None:
        f1, precision, recall = f1_direction_from_matrices(A, signed)
        metrics.update({"signed_f1": f1, "signed_precision": precision, "signed_recall": recall})
    return metrics


def _observation_dtype(config):
    training = diode_config(config).get("training", {})
    return _torch_dtype(training.get("observation_dtype", training.get("dtype", "float64")))


def _impute_with_diode(config_path, data, observed, mask, order):
    if mask is None:
        predictions, huber_delta = _fit_imputation_trajectory(
            config_path, data, observed, order, return_huber_delta=True,
        )
        mask = np.abs(observed - predictions) > huber_delta
    else:
        predictions = _fit_imputation_trajectory(config_path, data, observed, order)
    return impute_diode(observed, mask, data["pseudotime"], predictions)


def _fit_imputation_trajectory(config_path, data, observed, order, *, return_huber_delta=False):
    times = data["pseudotime"]
    dtype = data["Y_obs"].dtype
    y_obs = torch.tensor(observed[None, order, :], dtype=dtype)
    fit_data = {
        "Mask": data["Mask"],
        "Y_obs": y_obs,
        "X0": y_obs[:, 0, :],
        "t_span": torch.tensor(times[None, order], dtype=dtype),
        "pseudotime": times[order],
    }
    with TemporaryDirectory(prefix="diode-imputation-") as work_dir:
        with open(Path(work_dir) / "training.log", "w") as log, contextlib.redirect_stdout(log):
            sorted_predictions, huber_delta = _tune_diode(
                config_path, data=fit_data, fit_dir=Path(work_dir),
            )
    if not np.isfinite(sorted_predictions).all():
        raise ValueError(f"DIODE returned nonfinite trajectory predictions for {config_path}.")
    # The fit uses sorted pseudotime. Score in the original file order.
    ode_predictions = np.empty_like(sorted_predictions)
    ode_predictions[order] = sorted_predictions
    if return_huber_delta:
        return ode_predictions, huber_delta
    return ode_predictions


def _tune_diode(config_path, *, data=None, fit_dir=None):
    """Select DIODE hyperparameters by cross-validation and refit the winner."""
    start_time = time.perf_counter()
    config = load_config(config_path)
    root = resolve_config_repo_root(config, config_path)
    diode_cfg = diode_config(config)
    grid = diode_cfg.get("hyperparameter_grid")
    if not grid:
        raise ValueError("DIODE requires solvers.diode.hyperparameter_grid in the dataset config.")
    train_cfg = diode_cfg.get("training", {})
    tolerance_change = float(train_cfg.get("tolerance_change", 1e-4))
    dtype = _torch_dtype(train_cfg.get("dtype", "float64"))

    if data is None:
        data = load_ordered_expression_dataset(
            resolve_path(config["expression"], root),
            resolve_path(config["pseudotime"], root),
            resolve_path(config["genes"], root),
            reference_dir=optional_reference_dir(config, root),
            dtype=_observation_dtype(config),
            sort_kind=diode_cfg.get("sort_kind", "stable"),
        )

    device = _resolve_device(diode_cfg.get("device", "auto"))
    y_obs = data["Y_obs"].to(device=device, dtype=dtype)
    x0 = data["X0"].to(device=device, dtype=dtype)
    t_span = data["t_span"].to(device=device, dtype=dtype)
    # A structural parameter mask is shared by every CV model and the final
    # refit. Zeroing only the exported A would not constrain model training.
    mask = data["Mask"].to(device).clone()
    mask_diagonal = train_cfg.get("mask_diagonal", False)
    if not isinstance(mask_diagonal, bool):
        raise ValueError("training.mask_diagonal must be a YAML boolean (true or false).")
    if mask_diagonal:
        mask.fill_diagonal_(False)
    d = y_obs.shape[-1]

    train_mask, val_mask, split_df = _prepare_masks(diode_cfg, y_obs.shape[1], data["pseudotime"])
    cv_fold_masks, cv_fold_df = _build_cv_fold_masks(diode_cfg, train_mask, val_mask, data["pseudotime"], split_df)

    reference = data.get("reference_networks")
    unsigned = reference["unsigned"].to_numpy(dtype=int) if reference is not None else None
    signed = None
    if reference is not None and reference["signed"] is not None:
        signed = reference["signed"].to_numpy(dtype=int)
    edge_mask = _as_edge_mask(mask.cpu().numpy())

    output_dir = fit_dir if fit_dir is not None else resolve_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "training_network_mask.npy", mask.detach().cpu().numpy())
    np.savetxt(output_dir / "training_network_mask.csv", mask.detach().cpu().numpy(), delimiter=",", fmt="%d")
    run_log_dir = output_dir / "cv_logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(diode_cfg.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(diode_cfg.get("seed", 0)))

    x_star_init = y_obs[:, -1, :].clone()
    grid_df = _hyperparameter_grid(grid, tolerance_change)
    grid_df.to_csv(output_dir / "hyperparameter_grid.csv", index=False)
    split_df.to_csv(output_dir / "split.csv", index=False)
    cv_fold_df.to_csv(output_dir / "cv_folds.csv", index=False)

    fold_stop_rows = []
    combo_summary_rows = []
    all_history_parts = []

    for _, combo in grid_df.iterrows():
        combo_id = int(combo["combo_id"])
        run_name = str(combo["run_name"])
        huber_delta = float(combo["huber_delta"])
        l2_lambda = float(combo["l2_lambda"])
        print(f"[{combo_id:03d}/{len(grid_df):03d}] {run_name}")

        combo_A_vectors = []
        combo_fold_rows = []
        for fold_payload in cv_fold_masks:
            fold_id = int(fold_payload["fold_id"])
            fold_train_mask = fold_payload["train_mask_global"]
            model = _new_model(d, mask, x_star_init, dtype, device)
            experiment = train_lti_lbfgs_with_validation(
                model,
                X0=x0,
                t_span=t_span,
                Y_obs=y_obs,
                train_mask=fold_train_mask,
                val_mask=val_mask,
                A_true_list=[unsigned] if unsigned is not None else [],
                A_true_direction_list=[signed] if signed is not None else [],
                Mask_auroc=edge_mask,
                **_train_kwargs(
                    train_cfg,
                    huber_delta=huber_delta,
                    l2_lambda=l2_lambda,
                    tolerance_change=tolerance_change,
                    device=device,
                ),
            )

            history_df = experiment["history_df"].copy()
            history_df.insert(0, "run_name", run_name)
            history_df.insert(1, "combo_id", combo_id)
            history_df.insert(2, "fold_id", fold_id)
            history_df.insert(3, "huber_delta", huber_delta)
            history_df.insert(4, "l2_lambda", l2_lambda)
            history_df.insert(5, "tolerance_change", tolerance_change)
            history_df.insert(6, "cv_train_count", int(fold_train_mask.sum()))
            history_df.insert(7, "fixed_val_count", int(val_mask.sum()))
            history_df["train_loss"] = history_df["train_data_loss_post"]
            history_df["val_loss"] = history_df["val_data_loss_post"]
            history_df["train_mse"] = history_df["train_mse_post"]
            history_df["val_mse"] = history_df["val_mse_post"]
            history_df["train_mae"] = history_df["train_mae_post"]
            history_df["val_mae"] = history_df["val_mae_post"]
            all_history_parts.append(history_df)
            history_df.to_csv(run_log_dir / f"{run_name}_fold{fold_id}.csv", index=False)

            stop_row = extract_stop_row_from_history(history_df)
            A_eff_np = model.A_eff.detach().cpu().numpy()
            fold_row = {
                "run_name": run_name,
                "combo_id": combo_id,
                "fold_id": fold_id,
                "huber_delta": huber_delta,
                "l2_lambda": l2_lambda,
                "tolerance_change": tolerance_change,
                "epoch": int(stop_row["epoch"]),
                "stop_reason": str(stop_row["stop_reason"]),
                "train_loss": float(stop_row["train_data_loss_post"]),
                "val_loss": float(stop_row["val_data_loss_post"]),
                "train_mse": float(stop_row["train_mse_post"]),
                "val_mse": float(stop_row["val_mse_post"]),
                "train_rmse": float(stop_row["train_rmse_post"]),
                "val_rmse": float(stop_row["val_rmse_post"]),
                "train_mae": float(stop_row["train_mae_post"]),
                "val_mae": float(stop_row["val_mae_post"]),
                "delta_A_max_abs": float(stop_row["delta_A_max_abs"]),
                "step_max_abs": float(stop_row["lbfgs_state_step_max_abs"]),
                "grad_max_abs": float(stop_row["last_closure_grad_max_abs"]),
                "A_fro_norm": float(np.linalg.norm(A_eff_np)),
                "A_l1_norm": float(np.abs(A_eff_np).sum()),
                "A_nnz_abs_gt_1e-6": int((np.abs(A_eff_np) > float(train_cfg.get("a_nnz_threshold", 1e-6))).sum()),
            }
            if unsigned is not None:
                fold_row.update(
                    primary_auroc=float(stop_row["GS0_auroc_absA"]),
                    primary_f1=float(stop_row.get("GSDir0_f1", np.nan)),
                    primary_precision=float(stop_row.get("GSDir0_precision", np.nan)),
                    primary_recall=float(stop_row.get("GSDir0_recall", np.nan)),
                )
                if train_cfg.get("report_aupr", False):
                    fold_row["primary_aupr"] = float(stop_row["GS0_aupr_absA"])
            fold_stop_rows.append(fold_row)
            combo_fold_rows.append(fold_row)
            combo_A_vectors.append(flatten_masked_abs_edges(A_eff_np, edge_mask))

        stability = summarize_rank_stability(combo_A_vectors)
        pairwise_values = stability["pairwise_values"]
        combo_summary_rows.append({
            "run_name": run_name,
            "combo_id": combo_id,
            "huber_delta": huber_delta,
            "l2_lambda": l2_lambda,
            "tolerance_change": tolerance_change,
            "rank_stability_median": float(np.median(pairwise_values)) if pairwise_values.size else np.nan,
            "rank_stability_mean": float(np.mean(pairwise_values)) if pairwise_values.size else np.nan,
            "rank_stability_std": float(np.std(pairwise_values)) if pairwise_values.size else np.nan,
            "rank_stability_min": float(np.min(pairwise_values)) if pairwise_values.size else np.nan,
            "valid_fold_count": int(stability["valid_fold_count"]),
            "invalid_fold_count": int(stability["invalid_fold_count"]),
            "mean_stop_epoch": float(np.mean([row["epoch"] for row in combo_fold_rows])),
            "median_stop_epoch": float(np.median([row["epoch"] for row in combo_fold_rows])),
            "mean_train_loss": float(np.mean([row["train_loss"] for row in combo_fold_rows])),
            "median_train_loss": float(np.median([row["train_loss"] for row in combo_fold_rows])),
            "mean_val_loss": float(np.mean([row["val_loss"] for row in combo_fold_rows])),
            "median_val_loss": float(np.median([row["val_loss"] for row in combo_fold_rows])),
            "mean_train_mae": float(np.mean([row["train_mae"] for row in combo_fold_rows])),
            "median_train_mae": float(np.median([row["train_mae"] for row in combo_fold_rows])),
            "mean_val_mae": float(np.mean([row["val_mae"] for row in combo_fold_rows])),
            "median_val_mae": float(np.median([row["val_mae"] for row in combo_fold_rows])),
            "mean_train_mse": float(np.mean([row["train_mse"] for row in combo_fold_rows])),
            "median_train_mse": float(np.median([row["train_mse"] for row in combo_fold_rows])),
            "mean_val_mse": float(np.mean([row["val_mse"] for row in combo_fold_rows])),
            "median_val_mse": float(np.median([row["val_mse"] for row in combo_fold_rows])),
            "mean_A_fro_norm": float(np.mean([row["A_fro_norm"] for row in combo_fold_rows])),
            "median_A_fro_norm": float(np.median([row["A_fro_norm"] for row in combo_fold_rows])),
            "mean_A_nnz_abs_gt_1e-6": float(np.mean([row["A_nnz_abs_gt_1e-6"] for row in combo_fold_rows])),
            "median_A_nnz_abs_gt_1e-6": float(np.median([row["A_nnz_abs_gt_1e-6"] for row in combo_fold_rows])),
        })
        if unsigned is not None:
            aurocs = [row["primary_auroc"] for row in combo_fold_rows]
            f1_scores = [row["primary_f1"] for row in combo_fold_rows]
            combo_summary_rows[-1].update(
                mean_primary_auroc=float(np.mean(aurocs)),
                median_primary_auroc=float(np.median(aurocs)),
                mean_primary_f1=float(np.nanmean(f1_scores)),
                median_primary_f1=float(np.nanmedian(f1_scores)),
            )

    all_history_df = pd.concat(all_history_parts, ignore_index=True) if all_history_parts else pd.DataFrame()
    fold_stop_rows_df = pd.DataFrame(fold_stop_rows)
    combo_summary_df = pd.DataFrame(combo_summary_rows).sort_values(
        ["median_val_mae", "rank_stability_median", "rank_stability_mean", "rank_stability_std", "run_name"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    combo_summary_df.insert(0, "selection_rank", np.arange(1, len(combo_summary_df) + 1, dtype=int))
    combo_summary_df["selection_rule"] = "min_median_val_mae_then_max_median_rank_stability"

    all_history_df.to_csv(output_dir / "all_history.csv", index=False)
    fold_stop_rows_df.to_csv(output_dir / "stop_rows.csv", index=False)
    combo_summary_df.to_csv(output_dir / "combo_summary.csv", index=False)
    selected = combo_summary_df.iloc[0]
    pd.DataFrame([selected]).to_csv(output_dir / "selected_hyperparameter_combination.csv", index=False)

    selected_huber_delta = float(selected["huber_delta"])
    selected_l2_lambda = float(selected["l2_lambda"])
    full_train_mask = np.ones_like(train_mask, dtype=bool)
    final_eval_mask = np.asarray(val_mask, dtype=bool)

    torch.manual_seed(int(diode_cfg.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(diode_cfg.get("seed", 0)))
    if train_cfg.get("seed_all_rngs_on_refit", False):
        np.random.seed(int(diode_cfg.get("seed", 0)))
        random.seed(int(diode_cfg.get("seed", 0)))
    final_model = _new_model(d, mask, x_star_init, dtype, device)
    final_experiment = train_lti_lbfgs_with_validation(
        final_model,
        X0=x0,
        t_span=t_span,
        Y_obs=y_obs,
        train_mask=full_train_mask,
        val_mask=final_eval_mask,
        A_true_list=[unsigned] if unsigned is not None else [],
        A_true_direction_list=[signed] if signed is not None else [],
        Mask_auroc=edge_mask,
        **_train_kwargs(
            train_cfg,
            huber_delta=selected_huber_delta,
            l2_lambda=selected_l2_lambda,
            tolerance_change=tolerance_change,
            device=device,
        ),
    )

    final_history_df = final_experiment["history_df"].copy()
    final_history_df.insert(0, "run_name", str(selected["run_name"]))
    final_history_df.insert(1, "phase", "final_refit_100pct")
    final_history_df.insert(2, "train_mask_used", "full_data_train_mask_all_observations")
    final_history_df.insert(3, "eval_mask_used", "alternating_validation_set_for_reporting")
    final_history_df.insert(4, "train_observation_count", int(full_train_mask.sum()))
    final_history_df.insert(5, "eval_observation_count", int(final_eval_mask.sum()))
    final_history_df.to_csv(output_dir / "final_history.csv", index=False)

    final_stop_row = extract_stop_row_from_history(final_history_df)
    with torch.no_grad():
        A = final_model.A_eff.detach().cpu().numpy()
        X_star = final_model.X_star.detach().cpu().numpy()
        Y_pred = final_model(x0, t_span).detach().cpu().numpy()

    if fit_dir is not None:
        return Y_pred[0], selected_huber_delta

    save_model_outputs(
        output_dir,
        A=A,
        X0=x0.detach().cpu().numpy(),
        X_star=X_star,
        t_span=t_span.detach().cpu().numpy(),
        gene_names=data["gene_names"],
        stem="final_refit",
    )
    np.save(output_dir / "final_refit_Y_pred.npy", Y_pred)

    time_taken_seconds = time.perf_counter() - start_time
    metrics = {
        "dataset": config["dataset"],
        "expression_path": str(resolve_path(config["expression"], root)),
        "mask_diagonal": mask_diagonal,
        "run_name": str(selected["run_name"]),
        "selection_rule": str(selected["selection_rule"]),
        "selection_rank": int(selected["selection_rank"]),
        "huber_delta": selected_huber_delta,
        "l2_lambda": selected_l2_lambda,
        "tolerance_change": tolerance_change,
        "refit_train_fraction": 1.0,
        "train_observation_count": int(full_train_mask.sum()),
        "eval_observation_count": int(final_eval_mask.sum()),
        "epoch": int(final_stop_row["epoch"]),
        "stop_reason": str(final_stop_row["stop_reason"]),
        "train_loss": float(final_stop_row["train_data_loss_post"]),
        "val_loss": float(final_stop_row["val_data_loss_post"]),
        "train_mse": float(final_stop_row["train_mse_post"]),
        "val_mse": float(final_stop_row["val_mse_post"]),
        "train_mae": float(final_stop_row["train_mae_post"]),
        "val_mae": float(final_stop_row["val_mae_post"]),
        "A_fro_norm": float(np.linalg.norm(A)),
        "A_l1_norm": float(np.abs(A).sum()),
        "time_taken_seconds": time_taken_seconds,
    }

    perturb_scores, baseline_loss = edge_delta_val_loss_matrix(
        final_model,
        x0,
        t_span,
        y_obs,
        final_eval_mask,
        edge_mask,
        loss_type=train_cfg.get("loss_type", "huber"),
        huber_delta=selected_huber_delta,
        print_every=int(diode_cfg.get("perturbation_print_every", 500)),
    )
    np.savetxt(output_dir / "perturb_delta_val_loss_scores.csv", perturb_scores, delimiter=",")
    metrics["perturb_baseline_val_loss"] = baseline_loss
    evaluation_cfg = diode_cfg.get("evaluation", {})
    primary_score = evaluation_cfg.get("primary_score", "perturb")
    if primary_score not in ("absA", "perturb"):
        raise ValueError("evaluation.primary_score must be 'absA' or 'perturb'.")
    if unsigned is not None:
        metrics.update(_score_matrix(A, unsigned, signed, edge_mask))
        metrics["auroc_perturb"] = auroc_from_matrices(perturb_scores, unsigned, mask=edge_mask)
        metrics["auroc"] = metrics["auroc_absA" if primary_score == "absA" else "auroc_perturb"]
        if primary_score == "absA":
            # Preserve the source notebook's primary and supplementary metric names.
            metrics["primary_auroc"] = metrics["auroc_absA"]
            metrics["primary_aupr"] = metrics["aupr_absA"]
            metrics["primary_f1"] = metrics.get("signed_f1", np.nan)
            metrics["primary_precision"] = metrics.get("signed_precision", np.nan)
            metrics["primary_recall"] = metrics.get("signed_recall", np.nan)
            metrics["perturb_delta_val_loss_baseline"] = baseline_loss
            metrics["perturb_delta_val_loss_auroc"] = metrics["auroc_perturb"]
            metrics["perturb_delta_val_loss_aupr"] = aupr_from_matrices(perturb_scores, unsigned, mask=edge_mask)

    if evaluation_cfg.get("compute_sensitivity", False):
        metrics["sensitivity_status"] = "ok"
        metrics["sensitivity_error"] = ""
        try:
            sensitivity_scores, _, dominant_eigenvalue = dominant_eigenvalue_sensitivity_scores(A)
            if unsigned is not None:
                metrics["sensitivity_abs_auroc"] = auroc_from_matrices(sensitivity_scores, unsigned, mask=edge_mask)
                metrics["sensitivity_abs_aupr"] = aupr_from_matrices(sensitivity_scores, unsigned, mask=edge_mask)
        except ValueError as exc:
            metrics["sensitivity_status"] = "unstable"
            metrics["sensitivity_error"] = str(exc)
            sensitivity_scores = np.full_like(A, np.nan, dtype=float)
            dominant_eigenvalue = np.nan + 0j
            if unsigned is not None:
                metrics["sensitivity_abs_auroc"] = np.nan
                metrics["sensitivity_abs_aupr"] = np.nan
        np.savetxt(output_dir / "sensitivity_abs_scores.csv", sensitivity_scores, delimiter=",")
        metrics["sensitivity_dominant_eigenvalue_real"] = float(np.real(dominant_eigenvalue))
        metrics["sensitivity_dominant_eigenvalue_imag"] = float(np.imag(dominant_eigenvalue))

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "final_summary.csv", index=False)
    metrics_df.to_csv(output_dir / "evaluation_summary.csv", index=False)

    return {
        "config": config,
        "selection": selected.to_dict(),
        "cv_summary": combo_summary_df,
        "final_experiment": final_experiment,
        "metrics": metrics,
        "output_dir": output_dir,
    }
