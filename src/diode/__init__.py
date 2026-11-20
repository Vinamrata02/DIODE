"""Public interface for the DIODE package."""

from diode.experiment import DIODEBenchmark
from diode.diode import DIODE
from diode.solvers.diode_lbfgs import LTIExact_Steady


def run_dataset(*args, **kwargs):
    """Run one dataset config."""
    from diode.runners.run_dataset import run_dataset as _run_dataset

    return _run_dataset(*args, **kwargs)


def run_all(*args, **kwargs):
    """Run all selected dataset configs."""
    from diode.runners.run_all import run_all as _run_all

    return _run_all(*args, **kwargs)

__all__ = [
    "DIODEBenchmark",
    "DIODE",
    "LTIExact_Steady",
    "run_all",
    "run_dataset",
]

ExactLinearODE = DIODE
ExactLinearODEBenchmark = DIODEBenchmark
