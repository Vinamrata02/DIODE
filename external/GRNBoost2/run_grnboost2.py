"""Run GRNBoost2 on repository-format expression files, without a gold standard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diode.solvers.grnboost2 import run_grnboost2_from_numpy
from diode.utils.io import read_gene_names, save_model_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", required=True, help="Numeric genes-by-cells expression text file.")
    parser.add_argument("--genes", required=True, help="Gene names in expression-row order, one per line.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args(argv)
    gene_names = read_gene_names(args.genes)
    scores = run_grnboost2_from_numpy(
        np.loadtxt(args.expression, ndmin=2).T,
        gene_names=gene_names,
        seed=args.seed,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        output_dir=args.output_dir,
    )
    save_model_outputs(args.output_dir, A=scores, gene_names=gene_names, stem="scores")
    print(f"Saved GRNBoost2 scores to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
