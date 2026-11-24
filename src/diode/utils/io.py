"""Input/output helpers for datasets, model outputs, and reference networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def repo_root() -> Path:
    """Return the DIODE repository root."""
    return Path(__file__).resolve().parents[3]


def load_config(config_path) -> dict:
    """Load a YAML dataset config."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    return config


def diode_config(config: dict) -> dict:
    """Read solver-specific DIODE settings."""
    return (config.get("solvers") or {}).get("diode") or {}


def resolve_path(path, base_dir=None) -> Path:
    """Resolve a path relative to the repository root or a supplied base."""
    path = Path(path)
    if path.is_absolute():
        return path
    base = repo_root() if base_dir is None else Path(base_dir)
    return (base / path).resolve()


def read_gene_names(path) -> list[str]:
    """Read one gene name per line or a one-column text file."""
    values = np.loadtxt(path, dtype=str)
    if isinstance(values, str):
        return [values]
    return np.asarray(values, dtype=str).reshape(-1).tolist()


def optional_reference_dir(config, root):
    """Return the configured reference directory when it contains a reference."""
    directory = config.get("reference_network_dir")
    if directory:
        path = resolve_path(directory, root)
        if (path / "reference_network.csv").is_file():
            return path
    return None


def load_reference_network(reference_dir, gene_names=None):
    """Load unsigned and optional signed reference-network matrices."""
    reference_dir = Path(reference_dir)
    unsigned_path = reference_dir / "reference_network.csv"
    signed_path = reference_dir / "reference_network_signed.csv"

    if not unsigned_path.exists():
        raise FileNotFoundError(f"Missing reference network: {unsigned_path}")

    unsigned_df = _read_reference_matrix(unsigned_path, gene_names=gene_names, signed=False)
    signed_df = _read_reference_matrix(
        signed_path if signed_path.exists() else unsigned_path,
        gene_names=gene_names,
        signed=True,
    )

    if gene_names is not None:
        unsigned_df = unsigned_df.loc[gene_names, gene_names].copy()
        if signed_df is not None:
            signed_df = signed_df.loc[gene_names, gene_names].copy()
    else:
        unsigned_df = unsigned_df.copy()
        if signed_df is not None:
            signed_df = signed_df.copy()
    unsigned_values = unsigned_df.to_numpy(copy=True)
    np.fill_diagonal(unsigned_values, 0)
    unsigned_df = pd.DataFrame(unsigned_values, index=unsigned_df.index, columns=unsigned_df.columns)
    if signed_df is not None:
        signed_values = signed_df.to_numpy(copy=True)
        np.fill_diagonal(signed_values, 0)
        signed_df = pd.DataFrame(signed_values, index=signed_df.index, columns=signed_df.columns)

    return {
        "unsigned": unsigned_df,
        "signed": signed_df,
        "unsigned_path": unsigned_path,
        "signed_path": signed_path if signed_path.exists() else unsigned_path,
    }


def _read_reference_matrix(path, gene_names=None, *, signed=False):
    """Read a labeled CSV or a headerless numeric reference matrix."""
    path = Path(path)
    try:
        df = pd.read_csv(path, index_col=0)
        if gene_names is not None:
            gene_names = list(gene_names)
            if df.shape == (len(gene_names), len(gene_names)) and set(gene_names).issubset(df.index) and set(gene_names).issubset(df.columns):
                return df.loc[gene_names, gene_names]
        elif df.shape[0] == df.shape[1]:
            return df
    except Exception:
        pass

    try:
        matrix = np.loadtxt(path)
    except ValueError:
        matrix = np.loadtxt(path, delimiter=",")
    if gene_names is None:
        gene_names = [f"G{i}" for i in range(1, matrix.shape[0] + 1)]
    values = np.sign(matrix).astype(int) if signed else (matrix != 0).astype(int)
    return pd.DataFrame(values, index=gene_names, columns=gene_names)


def load_ordered_expression_dataset(
    expression_path,
    pseudotime_path,
    gene_names_path,
    *,
    reference_dir=None,
    dtype=torch.float32,
    sort_kind="stable",
):
    """Load genes-by-cells data. Use sort_kind=None to preserve file order."""
    expression = np.loadtxt(expression_path)
    pseudotime = np.loadtxt(pseudotime_path)
    gene_names = read_gene_names(gene_names_path)

    if expression.shape != (len(gene_names), pseudotime.size):
        raise ValueError(
            "Expected expression shape genes x cells "
            f"({len(gene_names)}, {pseudotime.size}); found {expression.shape}."
        )

    perm = np.argsort(pseudotime, kind=sort_kind) if sort_kind is not None else slice(None)
    time_sorted = pseudotime[perm]
    expression_sorted = expression[:, perm]
    expression_cells_by_genes = expression_sorted.T

    t_span = torch.tensor(time_sorted[None, :], dtype=dtype)
    y_obs = torch.tensor(expression_cells_by_genes[None, :, :], dtype=dtype)
    x0 = y_obs[0, 0, :][None, :]
    mask = torch.ones((len(gene_names), len(gene_names)), dtype=torch.bool)

    payload = {
        "expression_genes_by_cells": expression_sorted,
        "expression_cells_by_genes": expression_cells_by_genes,
        "pseudotime": time_sorted,
        "gene_names": gene_names,
        "t_span": t_span,
        "Y_obs": y_obs,
        "X0": x0,
        "Mask": mask,
    }

    if reference_dir is not None:
        payload["reference_networks"] = load_reference_network(reference_dir, gene_names=gene_names)

    return payload


def save_model_outputs(output_dir, *, A, X0=None, X_star=None, t_span=None, gene_names=None, stem="final_refit"):
    """Save learned model outputs in both NumPy/CSV-friendly formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / f"{stem}_A.csv", np.asarray(A), delimiter=",")
    np.save(output_dir / f"{stem}_A.npy", np.asarray(A))

    optional_arrays = {
        "X0": X0,
        "X_star": X_star,
        "t_span": t_span,
    }
    for name, value in optional_arrays.items():
        if value is not None:
            np.savetxt(output_dir / f"{stem}_{name}.csv", np.asarray(value), delimiter=",")
            np.save(output_dir / f"{stem}_{name}.npy", np.asarray(value))

    if gene_names is not None:
        pd.DataFrame({"gene": list(gene_names)}).to_csv(output_dir / f"{stem}_gene_order.csv", index=False)
