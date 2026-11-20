"""DIODE model and LBFGS training utilities."""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from diode.utils.evaluations import aupr_from_matrices, auroc_from_matrices, f1_direction_from_matrices

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
