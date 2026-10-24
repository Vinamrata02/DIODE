"""Public interface for the ExactLinearODE package."""

from exact_linear_ode.experiment import ExactLinearODEBenchmark
from exact_linear_ode.exact_linear_ode import ExactLinearODE
from exact_linear_ode.solvers.ours_lbfgs import LTIExact_Steady


def run_dataset(*args, **kwargs):
    """Run one dataset config."""
    from exact_linear_ode.runners.run_dataset import run_dataset as _run_dataset

    return _run_dataset(*args, **kwargs)


def run_all(*args, **kwargs):
    """Run all selected dataset configs."""
    from exact_linear_ode.runners.run_all import run_all as _run_all

    return _run_all(*args, **kwargs)

__all__ = [
    "ExactLinearODEBenchmark",
    "ExactLinearODE",
    "LTIExact_Steady",
    "run_all",
    "run_dataset",
]
