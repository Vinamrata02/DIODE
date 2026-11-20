"""Compatibility imports for the renamed DIODE package."""

import sys

import diode as _diode
import diode.experiment as _experiment
from diode import DIODE, DIODEBenchmark, LTIExact_Steady, run_all, run_dataset

__path__ = _diode.__path__
sys.modules[__name__ + ".experiment"] = _experiment
ExactLinearODE = DIODE
ExactLinearODEBenchmark = DIODEBenchmark

__all__ = ["DIODE", "DIODEBenchmark", "ExactLinearODE", "ExactLinearODEBenchmark",
           "LTIExact_Steady", "run_all", "run_dataset"]
