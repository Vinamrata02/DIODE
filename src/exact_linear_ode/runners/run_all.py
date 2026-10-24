"""Run all configured datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from exact_linear_ode.runners.run_dataset import run_dataset


DEFAULT_CONFIGS = [
    "configs/data1.yaml",
    "configs/data2.yaml",
    "configs/data3.yaml",
    "configs/data4_epiblast.yaml",
    "configs/data4_trophectoderm.yaml",
]


def run_all(configs=None, repo_root=None):
    """Run all supplied dataset configs and return a combined metrics table."""
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    configs = DEFAULT_CONFIGS if configs is None else configs

    rows = []
    for config in configs:
        result = run_dataset(repo_root / config)
        rows.append(result["metrics"])

    combined = pd.DataFrame(rows)
    output_path = repo_root / "outputs" / "all_evaluation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all ExactLinearODE datasets.")
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("configs", nargs="*", help="Optional config paths relative to repo root.")
    args = parser.parse_args(argv)
    combined = run_all(configs=args.configs or None, repo_root=args.repo_root)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
