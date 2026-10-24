"""Evaluation helpers for ExactLinearODE experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

def corr_masked(A: torch.Tensor,
                B: torch.Tensor,
                mask: torch.Tensor,
                *,
                exclude_diag: bool = True,
                return_tensor: bool = False) -> float | torch.Tensor:
    """
    Pearson correlation between entries of A and B selected by `mask`.
    Optionally excludes diagonal entries.

    Args:
        A, B        : [d, d] tensors (device/dtype can differ; will be aligned to A)
        mask        : [d, d] boolean tensor indicating which entries to use
        exclude_diag: if True, remove diagonal entries from consideration
        return_tensor: if True, return a 0-dim tensor; else return Python float

    Returns:
        r (float or 0-dim tensor): Pearson correlation over masked entries.
                                   Returns NaN if <2 points or zero variance.
    """
    # Align devices/dtypes
    device, dtype = A.device, A.dtype
    A = A.detach().to(device=device, dtype=dtype)
    B = B.detach().to(device=device, dtype=dtype)
    M = mask.to(device=device, dtype=torch.bool)

    if exclude_diag:
        d = M.shape[0]
        M = M & ~torch.eye(d, dtype=torch.bool, device=device)

    # Select entries
    a = A[M].flatten()
    b = B[M].flatten()

    # Guards: need at least 2 points and non-zero std
    if a.numel() < 2:
        r = torch.tensor(float('nan'), device=device, dtype=dtype)
    else:
        a_std = a.std(unbiased=False)
        b_std = b.std(unbiased=False)
        if (a_std == 0) or (b_std == 0):
            r = torch.tensor(float('nan'), device=device, dtype=dtype)
        else:
            # Pearson r = cov / (std_a * std_b), using population moments
            cov = ((a - a.mean()) * (b - b.mean())).mean()
            r = cov / (a_std * b_std)

    return r if return_tensor else float(r)


def corr2d_overall(A: np.ndarray, B: np.ndarray, mask: np.ndarray=None, nan_policy: str="raise") -> float:
    """
    Pearson correlation between ALL paired entries of A and B (flattened).
    - mask: optional boolean array same shape as A/B to select entries.
    - nan_policy: 'raise' | 'omit' (ignore NaN pairs).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shapes must match, got {A.shape} vs {B.shape}")
    
    m = np.ones_like(A, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != A.shape: 
        raise ValueError("mask must have same shape as A/B")
    
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy must be 'raise' or 'omit'")

    if nan_policy == "omit":
        m &= np.isfinite(A) & np.isfinite(B)
    elif np.isnan(A).any() or np.isnan(B).any():
        raise ValueError("NaNs present; set nan_policy='omit' to ignore them")

    a = A[m].ravel()
    b = B[m].ravel()
    if a.size < 2:
        return np.nan
    a_std = a.std(ddof=0)
    b_std = b.std(ddof=0)
    if a_std == 0 or b_std == 0:
        return np.nan
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return cov / (a_std * b_std)


def auroc_from_matrices(A_pre, A_true, *, mask=None):
    A_pre  = np.asarray(A_pre, dtype=float)
    A_true = np.asarray(A_true, dtype=int)

    if A_pre.shape != A_true.shape:
        raise ValueError(f"shape mismatch: {A_pre.shape} vs {A_true.shape}")

    # Optional: apply a boolean mask (e.g., exclude diagonal)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        A_pre  = A_pre[mask]
        A_true = A_true[mask]
    else:
        A_pre  = A_pre.ravel()
        A_true = A_true.ravel()

    # AUROC needs at least one positive and one negative
    if A_true.min() == A_true.max():
        raise ValueError("AUROC undefined: ground truth has only one class.")

    return roc_auc_score(A_true, A_pre)


def f1_direction_from_matrices(A_pre, A_true):
    A_pre  = np.asarray(A_pre, dtype=float)
    A_true = np.asarray(A_true, dtype=int)

    if A_pre.shape != A_true.shape:
        raise ValueError(f"shape mismatch: {A_pre.shape} vs {A_true.shape}")

    A_pre_1d = A_pre[A_true!=0] 
    A_pre_1d_bin = np.where(A_pre_1d > 0, 1, -1)
    A_true_1d = A_true[A_true!=0]

    # AUROC needs at least one positive and one negative
    if A_true.min() == A_true.max():
        raise ValueError("AUROC undefined: ground truth has only one class.")

    return (
        f1_score(A_true_1d, A_pre_1d_bin, pos_label=1, zero_division=0),
        precision_score(A_true_1d, A_pre_1d_bin, pos_label=1, zero_division=0),
        recall_score(A_true_1d, A_pre_1d_bin, pos_label=1, zero_division=0),
    )


def auroc_matrix_by_row(A_pre, A_true, TopK=10):
    
    auroc_rows = []
    for rr in range(len(A_true)):
        if np.sum(A_true[rr,:]) < 1:
            auroc_rows.extend([0])
            continue
        
        idx = np.argpartition(np.abs(A_pre[rr,:]), -TopK)[-TopK:]

        # make an array of zeros and keep only those k positions
        y = np.zeros_like(np.abs(A_pre[rr,:]))
        y[idx] = np.abs(A_pre[rr,:])[idx]
        
        auroc_tmp = roc_auc_score(A_true[rr,:], y)
        auroc_rows.extend([auroc_tmp])
        
    return auroc_rows


def aupr_from_matrices(A_pre, A_true, *, mask=None):
    A_pre  = np.asarray(A_pre, dtype=float)
    A_true = np.asarray(A_true, dtype=int)

    if A_pre.shape != A_true.shape:
        raise ValueError(f"shape mismatch: {A_pre.shape} vs {A_true.shape}")

    # Optional mask (e.g., exclude diagonal)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        A_pre  = A_pre[mask]
        A_true = A_true[mask]
    else:
        A_pre  = A_pre.ravel()
        A_true = A_true.ravel()

    # Need at least one positive and one negative
    uniq = np.unique(A_true)
    if uniq.size < 2:
        raise ValueError("AUPR undefined: ground truth has only one class.")

    # Precision–Recall curve
    precision, recall, _ = precision_recall_curve(A_true, A_pre)

    # area under PR curve (precision as a function of recall)
    aupr = auc(recall, precision)

    return float(aupr)


def aupr_from_matrices_by_row(A_pre, A_true, *, mask=None):
    A_pre  = np.asarray(A_pre, dtype=float)
    A_true = np.asarray(A_true, dtype=int)

    if A_pre.shape != A_true.shape:
        raise ValueError(f"shape mismatch: {A_pre.shape} vs {A_true.shape}")
        
    nrow, ncol = A_pre.shape
    Aupr_Rows = np.zeros(nrow)
    for i in range(nrow):
        # Optional mask (e.g., exclude diagonal)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            A_pre_r  = A_pre[i,mask[i,:]]
            A_true_r = A_true[i,mask[i,:]]
        else:
            A_pre_r  = A_pre[i,:].ravel()
            A_true_r = A_true[i,:].ravel()

        # Need at least one positive and one negative
        uniq = np.unique(A_true_r)
        if uniq.size < 2:
            continue
            raise ValueError("AUPR undefined: ground truth has only one class.")

        # Precision–Recall curve
        precision, recall, _ = precision_recall_curve(A_true_r, A_pre_r)

        # area under PR curve (precision as a function of recall)
        aupr = auc(recall, precision)
        
        Aupr_Rows[i] = aupr

    return float(np.mean(Aupr_Rows)), Aupr_Rows

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

def infer_pseudotime(x):
    x = np.asarray(x, dtype=float)
    latent = PCA(n_components=min(20, x.shape[0] - 1, x.shape[1]), random_state=0).fit_transform(x)
    return latent[:, 0]


def pos_score(pseudotime, true_order):
    true_order = np.asarray(true_order, dtype=float)
    denom = np.triu(np.abs(true_order[:, None] - true_order[None, :])).sum()
    ordered = true_order[np.argsort(np.asarray(pseudotime, dtype=float))]
    numer = np.triu(ordered[None, :] - ordered[:, None]).sum()
    return float(abs(numer / denom)) if denom > 0 else np.nan


def wasserstein_1(x, y):
    n = min(len(x), len(y))
    cost = cdist(np.asarray(x[:n], dtype=float), np.asarray(y[:n], dtype=float), metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())

def signed_call_from_weights(values, top_k=None, abs_threshold=None):
    values = np.asarray(values, dtype=float)
    signed = np.zeros(values.shape, dtype=int)
    idx = np.flatnonzero(np.isfinite(values) & (np.abs(values) > 0))
    if abs_threshold is not None:
        idx = idx[np.abs(values[idx]) >= abs_threshold]
    if top_k is not None and idx.size > top_k:
        strongest = np.argsort(np.abs(values[idx]))[-top_k:]
        idx = idx[np.sort(strongest)]
    signed[idx] = np.sign(values[idx]).astype(int)
    return signed


def signed_f1_stats(gt_signed, pred_signed):
    gt_signed = np.asarray(gt_signed, dtype=int)
    pred_signed = np.asarray(pred_signed, dtype=int)
    correct = (gt_signed != 0) & (pred_signed == gt_signed)
    tp = int(correct.sum())
    fp = int(((pred_signed != 0) & ~correct).sum())
    fn = int(((gt_signed != 0) & ~correct).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
