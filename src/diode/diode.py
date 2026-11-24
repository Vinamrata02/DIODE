"""User-facing DIODE wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diode.runners.run_dataset import _resolve_repo_root, run_dataset
from diode.utils.evaluations import auroc_from_matrices
from diode.utils.io import load_config, load_reference_network, optional_reference_dir


class DIODE:
    """Run DIODE from a dataset config."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.result_ = None
        self.output_dir_ = None
        self.metrics_ = None

    def fit(self):
        """Run the configured training workflow and return the runner result dict."""
        self.result_ = run_dataset(self.config_path)
        self.output_dir_ = Path(self.result_["output_dir"])
        self.metrics_ = self.result_["metrics"]
        return self.result_

    def evaluate(self, A_true=None, mask=None):
        """Evaluate saved perturbation scores using a supplied or configured reference."""
        reference_dir = None
        if A_true is None:
            config = load_config(self.config_path)
            root = _resolve_repo_root(config, self.config_path)
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
