"""User-facing ExactLinearODE wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from exact_linear_ode.runners.run_dataset import run_dataset
from exact_linear_ode.utils.evaluations import auroc_from_matrices


class ExactLinearODE:
    """Run ExactLinearODE from a dataset config."""

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

    def evaluate(self, A_true, mask=None):
        """Evaluate saved perturbation scores against a ground-truth adjacency matrix."""
        if self.output_dir_ is None:
            if self.result_ is None:
                raise RuntimeError("Call fit() before evaluate().")
            self.output_dir_ = Path(self.result_["output_dir"])

        score_path = self.output_dir_ / "perturb_delta_val_loss_scores.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing perturbation score file: {score_path}")

        scores = np.loadtxt(score_path, delimiter=",")
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
