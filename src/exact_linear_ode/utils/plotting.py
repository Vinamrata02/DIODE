"""Plotting helpers shared by notebooks and runners."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def plot_fit(t_span, Y_obs, Y_pred, b: int = 0, mask: torch.Tensor | None = None, title_prefix=""):
    """
    t_span: [T] tensor
    Y_obs, Y_pred: [T,d] or [B,T,d]
    b: which trajectory to plot if batch dimension is present
    mask: optional boolean mask [T] or [B,T]; True = observed
    """
    # to numpy
    t = t_span.detach().cpu().numpy()

    # normalize shapes to [T,d]
    if Y_obs.ndim == 3:
        Yo = Y_obs[b].detach().cpu().numpy()   # [T,d]
        Yp = Y_pred[b].detach().cpu().numpy()
        m  = None if mask is None else (mask[b].detach().cpu().numpy().astype(bool))
    else:
        Yo = Y_obs.detach().cpu().numpy()
        Yp = Y_pred.detach().cpu().numpy()
        m  = None if mask is None else (mask.detach().cpu().numpy().astype(bool))

    T, d = Yo.shape
    assert Yp.shape == (T, d), f"Shape mismatch: Y_pred {Yp.shape} vs Y_obs {Yo.shape}"

    # default mask = all observed
    if m is None:
        m = np.ones(T, dtype=bool)

    # one figure per state dimension; multiple lines on each figure
    for j in range(d):
        plt.figure()
        # observed points (masked)
        plt.plot(t[m], Yo[m, j], label="Y_obs")
        # prediction (full curve)
        plt.plot(t, Yp[:, j], label="Y_pred")
        plt.xlabel("t")
        plt.ylabel(f"state[{j}]")
        plt.title(f"{title_prefix} trajectory (dim {j})")
        plt.legend()
        plt.show()

def plot_pseudotime_split(split_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    split_colors = {'train': '#1f77b4', 'validation': '#d62728'}
    for split_name, group in split_df.groupby('split', sort=False):
        axes[0].scatter(
            group['obs_index'],
            group['pseudotime'],
            s=28,
            alpha=0.85,
            color=split_colors[split_name],
            label=split_name,
        )
    axes[0].set_xlabel('Observation index (sorted pseudotime order)')
    axes[0].set_ylabel('Pseudotime')
    axes[0].set_title('Train/validation split across pseudotime')
    axes[0].legend()

    split_counts = (
        split_df.groupby(['bin_id', 'split'])['obs_index']
        .count()
        .unstack(fill_value=0)
        .reindex(columns=['train', 'validation'], fill_value=0)
    )
    split_counts.plot(kind='bar', stacked=True, ax=axes[1], color=[split_colors['train'], split_colors['validation']])
    axes[1].set_xlabel('Pseudotime bin')
    axes[1].set_ylabel('Number of observations')
    axes[1].set_title('Bin-wise split counts')
    axes[1].legend(title='Split')

    plt.tight_layout()
    plt.show()


def plot_selection_diagnostics(history_df, selection_summary_df, stop_tolerance_grad, stop_tolerance_change):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(history_df['epoch'], history_df['train_data_loss_post'], label='Training data loss', linewidth=2)
    ax.plot(history_df['epoch'], history_df['val_data_loss_post'], label='Validation data loss', linewidth=2)
    best_val_row = history_df.loc[history_df['val_data_loss_post'].idxmin()]
    ax.scatter(best_val_row['epoch'], best_val_row['val_data_loss_post'], s=80, color='black', label='Min validation loss', zorder=5)
    ax.set_xlabel('Outer epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Training and validation loss')

    for criterion, color in [
        ('first_meets_tolerance_grad', '#2ca02c'),
        ('first_meets_tolerance_change_step', '#9467bd'),
        ('first_meets_tolerance_change_loss', '#8c564b'),
        ('first_meets_delta_A_norm', '#e377c2'),
        ('first_meets_delta_A_max_abs', '#d62728'),
        ('first_meets_delta_X_star_norm', '#7f7f7f'),
        ('first_meets_delta_X_star_max_abs', '#17becf'),
    ]:
        match = selection_summary_df.loc[selection_summary_df['criterion'] == criterion]
        if not match.empty and pd.notna(match.iloc[0]['epoch']):
            epoch = int(match.iloc[0]['epoch'])
            ax.axvline(epoch, linestyle='--', linewidth=1.5, color=color, label=criterion)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(history_df['epoch'], history_df['last_closure_grad_max_abs'], label='Last closure max|grad|', linewidth=2)
    ax.plot(history_df['epoch'], history_df['post_step_train_grad_max_abs'], label='Post-step recomputed max|grad|', linewidth=1.5)
    ax.axhline(stop_tolerance_grad, linestyle='--', color='black', label=f'stop_tolerance_grad={stop_tolerance_grad:.1e}')
    ax.set_xlabel('Outer epoch')
    ax.set_ylabel('Gradient magnitude')
    ax.set_yscale('log')
    ax.set_title('Gradient-based stopping quantities')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(history_df['epoch'], history_df['lbfgs_state_step_max_abs'], label='max|d * t|', linewidth=2)
    ax.plot(history_df['epoch'], history_df['lbfgs_state_loss_change_abs'], label='|loss - prev_loss|', linewidth=2)
    ax.axhline(stop_tolerance_change, linestyle='--', color='black', label=f'stop_tolerance_change={stop_tolerance_change:.1e}')
    ax.set_xlabel('Outer epoch')
    ax.set_ylabel('Tolerance-change quantity')
    ax.set_yscale('log')
    ax.set_title('LBFGS tolerance_change diagnostics')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(history_df['epoch'], history_df['delta_A_fro_norm'], label='||ΔA||_F', linewidth=2)
    ax.plot(history_df['epoch'], history_df['delta_A_max_abs'], label='max|ΔA|', linewidth=1.8)
    ax.plot(history_df['epoch'], history_df['delta_X_star_l2_norm'], label='||Δx*||_2', linewidth=2)
    ax.plot(history_df['epoch'], history_df['delta_X_star_max_abs'], label='max|Δx*|', linewidth=1.8)
    ax.axhline(stop_tolerance_change, linestyle='--', color='black', label=f'stop_tolerance_change={stop_tolerance_change:.1e}')
    ax.set_xlabel('Outer epoch')
    ax.set_ylabel('Parameter change')
    ax.set_yscale('log')
    ax.set_title('Outer-epoch parameter changes')
    ax.legend()

    plt.tight_layout()
    plt.show()

def maybe_set_log_scale(ax, values, axis='x', ratio_threshold=50.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size < 2:
        return
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin <= 0:
        return
    if vmax / vmin >= ratio_threshold:
        if axis == 'x':
            ax.set_xscale('log')
        else:
            ax.set_yscale('log')


def plot_loss_auroc_relationships(all_history_df, stop_rows_df, output_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # train vs val, colored by AUROC
    ax = axes[0]
    sc = ax.scatter(
        all_history_df['train_data_loss_post'],
        all_history_df['val_data_loss_post'],
        c=all_history_df['GS0_auroc_absA'],
        cmap='viridis',
        s=8,
        alpha=0.2,
        edgecolors='none',
    )
    ax.scatter(
        stop_rows_df['train_loss'],
        stop_rows_df['val_loss'],
        c=stop_rows_df['primary_auroc'],
        cmap='viridis',
        s=36,
        edgecolors='black',
        linewidths=0.3,
    )
    ax.set_xlabel('Training loss')
    ax.set_ylabel('Validation loss')
    ax.set_title('Train vs validation loss')
    maybe_set_log_scale(ax, all_history_df['train_data_loss_post'], axis='x')
    maybe_set_log_scale(ax, all_history_df['val_data_loss_post'], axis='y')

    # train vs AUROC
    ax = axes[1]
    ax.scatter(all_history_df['train_data_loss_post'], all_history_df['GS0_auroc_absA'], s=8, alpha=0.2, color='#1f77b4', edgecolors='none')
    ax.scatter(stop_rows_df['train_loss'], stop_rows_df['primary_auroc'], s=36, color='#d62728', edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Training loss')
    ax.set_ylabel('Primary AUROC')
    ax.set_title('Training loss vs AUROC')
    maybe_set_log_scale(ax, all_history_df['train_data_loss_post'], axis='x')

    # validation vs AUROC
    ax = axes[2]
    ax.scatter(all_history_df['val_data_loss_post'], all_history_df['GS0_auroc_absA'], s=8, alpha=0.2, color='#2ca02c', edgecolors='none')
    ax.scatter(stop_rows_df['val_loss'], stop_rows_df['primary_auroc'], s=36, color='#ff7f0e', edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Validation loss')
    ax.set_ylabel('Primary AUROC')
    ax.set_title('Validation loss vs AUROC')
    maybe_set_log_scale(ax, all_history_df['val_data_loss_post'], axis='x')

    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('Primary AUROC')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_loss_error_metric_relationships(all_history_df, stop_rows_df, output_path=None, correlation_csv=None):
    specs = [
        ('train_data_loss_post', 'train_mae_post', 'train_loss', 'train_mae', 'Training Huber loss', 'Training MAE'),
        ('val_data_loss_post', 'val_mae_post', 'val_loss', 'val_mae', 'Validation Huber loss', 'Validation MAE'),
        ('train_data_loss_post', 'train_mse_post', 'train_loss', 'train_mse', 'Training Huber loss', 'Training MSE'),
        ('val_data_loss_post', 'val_mse_post', 'val_loss', 'val_mse', 'Validation Huber loss', 'Validation MSE'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.reshape(-1)
    correlation_rows = []

    for ax, (hist_x, hist_y, stop_x, stop_y, xlabel, ylabel) in zip(axes, specs):
        hx, hy = _finite_pair_arrays(all_history_df, hist_x, hist_y)
        if hx.size:
            ax.scatter(hx, hy, s=8, alpha=0.18, color='#1f77b4', edgecolors='none', label='all epochs')
        sx, sy = _finite_pair_arrays(stop_rows_df, stop_x, stop_y)
        if sx.size:
            ax.scatter(sx, sy, s=34, alpha=0.85, color='#ff7f0e', edgecolors='black', linewidths=0.3, label='stop rows')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')
        ax.grid(alpha=0.2)
        maybe_set_log_scale(ax, hx, axis='x')
        maybe_set_log_scale(ax, hy, axis='y')
        correlation_rows.append(_correlation_record(all_history_df, 'all_epochs', hist_x, hist_y))
        correlation_rows.append(_correlation_record(stop_rows_df, 'stop_rows', stop_x, stop_y))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    correlation_df = pd.DataFrame(correlation_rows)
    if correlation_csv is not None:
        correlation_df.to_csv(correlation_csv, index=False)
    display(correlation_df)
    return correlation_df


def plot_error_auroc_relationships(all_history_df, stop_rows_df, output_path=None, correlation_csv=None):
    specs = [
        ('train_mae_post', 'GS0_auroc_absA', 'train_mae', 'primary_auroc', 'Training MAE', 'Primary AUROC'),
        ('val_mae_post', 'GS0_auroc_absA', 'val_mae', 'primary_auroc', 'Validation MAE', 'Primary AUROC'),
        ('train_mse_post', 'GS0_auroc_absA', 'train_mse', 'primary_auroc', 'Training MSE', 'Primary AUROC'),
        ('val_mse_post', 'GS0_auroc_absA', 'val_mse', 'primary_auroc', 'Validation MSE', 'Primary AUROC'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.reshape(-1)
    correlation_rows = []

    for ax, (hist_x, hist_y, stop_x, stop_y, xlabel, ylabel) in zip(axes, specs):
        hx, hy = _finite_pair_arrays(all_history_df, hist_x, hist_y)
        if hx.size:
            ax.scatter(hx, hy, s=8, alpha=0.18, color='#1f77b4', edgecolors='none', label='all epochs')
        sx, sy = _finite_pair_arrays(stop_rows_df, stop_x, stop_y)
        if sx.size:
            ax.scatter(sx, sy, s=36, alpha=0.9, color='#d62728', edgecolors='black', linewidths=0.3, label='stop rows')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs AUROC')
        maybe_set_log_scale(ax, hx if hx.size else sx, axis='x')
        ax.grid(alpha=0.2)

        correlation_rows.append(_correlation_record(all_history_df, 'all_epochs', hist_x, hist_y))
        correlation_rows.append(_correlation_record(stop_rows_df, 'stop_rows', stop_x, stop_y))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    correlation_df = pd.DataFrame(correlation_rows)
    if correlation_csv is not None:
        correlation_df.to_csv(correlation_csv, index=False)
    return correlation_df
