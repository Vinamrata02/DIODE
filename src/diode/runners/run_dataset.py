"""Run one dataset using a config file.

This runner mirrors the notebook workflow:
1. load data and reference networks
2. build the fixed train/validation split
3. run hyperparameter selection across CV folds on the training partition
4. refit the selected hyperparameters on all observations
5. report final AUROC/AUPR/signed metrics from the full-data refit
"""

from __future__ import annotations

import argparse
import random
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diode.solvers.diode_lbfgs import (
    LTIExact_Steady,
    edge_delta_val_loss_matrix,
    extract_stop_row_from_history,
    flatten_masked_abs_edges,
    format_hparam,
    stratified_pseudotime_kfold_split,
    summarize_rank_stability,
    train_lti_lbfgs_with_validation,
)
from diode.utils.evaluations import (
    aupr_from_matrices,
    auroc_from_matrices,
    dominant_eigenvalue_sensitivity_scores,
    f1_direction_from_matrices,
)
from diode.utils.io import load_config, load_ordered_expression_dataset, resolve_path, save_model_outputs


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


def _resolve_repo_root(config, config_path):
    config_path = Path(config_path).resolve()
    if config.get("repo_root"):
        return Path(config["repo_root"]).expanduser().resolve()
    if config_path.parent.name == "configs":
        return config_path.parents[1]
    return Path.cwd().resolve()


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


def _hyperparameter_grid(config, train_cfg):
    grid = config.get("hyperparameter_grid") or {
        "huber_delta": [float(train_cfg.get("huber_delta", 2.5))],
        "l2_lambda": [float(train_cfg.get("l2_lambda", 1e-7))],
        "tolerance_change": [float(train_cfg.get("stop_tolerance_change", 1e-5))],
    }
    rows = list(product(grid["huber_delta"], grid["l2_lambda"], grid["tolerance_change"]))
    df = pd.DataFrame(rows, columns=["huber_delta", "l2_lambda", "tolerance_change"])
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


def run_dataset(config_path):
    """Run one configured dataset and return a summary dictionary."""
    start_time = time.perf_counter()
    config = load_config(config_path)
    root = _resolve_repo_root(config, config_path)
    train_cfg = config.get("training", {})
    dtype = _torch_dtype(train_cfg.get("dtype", "float64"))

    data = load_ordered_expression_dataset(
        resolve_path(config["expression"], root),
        resolve_path(config["pseudotime"], root),
        resolve_path(config["genes"], root),
        reference_dir=resolve_path(config["reference_network_dir"], root),
        dtype=_torch_dtype(train_cfg.get("observation_dtype", train_cfg.get("dtype", "float64"))),
        sort_kind=config.get("sort_kind", "stable"),
    )

    device = _resolve_device(config.get("device", "auto"))
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

    train_mask, val_mask, split_df = _prepare_masks(config, y_obs.shape[1], data["pseudotime"])
    cv_fold_masks, cv_fold_df = _build_cv_fold_masks(config, train_mask, val_mask, data["pseudotime"], split_df)

    reference = data["reference_networks"]
    unsigned = reference["unsigned"].to_numpy(dtype=int)
    signed = reference["signed"].to_numpy(dtype=int) if reference["signed"] is not None else None
    edge_mask = _as_edge_mask(mask.cpu().numpy())

    output_dir = resolve_path(config["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "training_network_mask.npy", mask.detach().cpu().numpy())
    np.savetxt(output_dir / "training_network_mask.csv", mask.detach().cpu().numpy(), delimiter=",", fmt="%d")
    run_log_dir = output_dir / "cv_logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.get("seed", 0)))

    x_star_init = y_obs[:, -1, :].clone()
    grid_df = _hyperparameter_grid(config, train_cfg)
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
        tolerance_change = float(combo["tolerance_change"])
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
                A_true_list=[unsigned],
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
                "primary_auroc": float(stop_row["GS0_auroc_absA"]),
                "primary_f1": float(stop_row.get("GSDir0_f1", np.nan)),
                "primary_precision": float(stop_row.get("GSDir0_precision", np.nan)),
                "primary_recall": float(stop_row.get("GSDir0_recall", np.nan)),
            }
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
            "mean_primary_auroc": float(np.mean([row["primary_auroc"] for row in combo_fold_rows])),
            "median_primary_auroc": float(np.median([row["primary_auroc"] for row in combo_fold_rows])),
            "mean_primary_f1": float(np.nanmean([row["primary_f1"] for row in combo_fold_rows])),
            "median_primary_f1": float(np.nanmedian([row["primary_f1"] for row in combo_fold_rows])),
        })

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
    selected_tolerance_change = float(selected["tolerance_change"])
    full_train_mask = np.ones_like(train_mask, dtype=bool)
    final_eval_mask = np.asarray(val_mask, dtype=bool)

    torch.manual_seed(int(config.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.get("seed", 0)))
    if train_cfg.get("seed_all_rngs_on_refit", False):
        np.random.seed(int(config.get("seed", 0)))
        random.seed(int(config.get("seed", 0)))
    final_model = _new_model(d, mask, x_star_init, dtype, device)
    final_experiment = train_lti_lbfgs_with_validation(
        final_model,
        X0=x0,
        t_span=t_span,
        Y_obs=y_obs,
        train_mask=full_train_mask,
        val_mask=final_eval_mask,
        A_true_list=[unsigned],
        A_true_direction_list=[signed] if signed is not None else [],
        Mask_auroc=edge_mask,
        **_train_kwargs(
            train_cfg,
            huber_delta=selected_huber_delta,
            l2_lambda=selected_l2_lambda,
            tolerance_change=selected_tolerance_change,
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

    final_scores = _score_matrix(A, unsigned, signed, edge_mask)
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
        "tolerance_change": selected_tolerance_change,
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
        **final_scores,
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
        print_every=int(config.get("perturbation_print_every", 500)),
    )
    np.savetxt(output_dir / "perturb_delta_val_loss_scores.csv", perturb_scores, delimiter=",")
    metrics["perturb_baseline_val_loss"] = baseline_loss
    metrics["auroc_perturb"] = auroc_from_matrices(perturb_scores, unsigned, mask=edge_mask)
    evaluation_cfg = config.get("evaluation", {})
    primary_score = evaluation_cfg.get("primary_score", "perturb")
    if primary_score not in ("absA", "perturb"):
        raise ValueError("evaluation.primary_score must be 'absA' or 'perturb'.")
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
            metrics["sensitivity_abs_auroc"] = auroc_from_matrices(sensitivity_scores, unsigned, mask=edge_mask)
            metrics["sensitivity_abs_aupr"] = aupr_from_matrices(sensitivity_scores, unsigned, mask=edge_mask)
        except ValueError as exc:
            metrics["sensitivity_status"] = "unstable"
            metrics["sensitivity_error"] = str(exc)
            sensitivity_scores = np.full_like(A, np.nan, dtype=float)
            dominant_eigenvalue = np.nan + 0j
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one DIODE dataset.")
    parser.add_argument("config", help="Path to a YAML config.")
    args = parser.parse_args(argv)
    result = run_dataset(args.config)
    print(pd.DataFrame([result["metrics"]]).to_string(index=False))


if __name__ == "__main__":
    main()
