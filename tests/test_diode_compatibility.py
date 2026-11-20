import unittest

from diode import DIODE, DIODEBenchmark, LTIExact_Steady
from exact_linear_ode import ExactLinearODE, ExactLinearODEBenchmark
from exact_linear_ode.exact_linear_ode import ExactLinearODE as LegacyModel
from exact_linear_ode.experiment import ExactLinearODEBenchmark as LegacyBenchmark
from exact_linear_ode.solvers.ours_lbfgs import LTIExact_Steady as LegacySteady


class DIODECompatibilityTests(unittest.TestCase):
    def test_existing_imports_keep_the_same_model_and_benchmark(self):
        self.assertIs(ExactLinearODE, DIODE)
        self.assertIs(LegacyModel, DIODE)
        self.assertIs(ExactLinearODEBenchmark, DIODEBenchmark)
        self.assertIs(LegacyBenchmark, DIODEBenchmark)
        self.assertIs(LegacySteady, LTIExact_Steady)

    def test_existing_solver_inputs_select_diode(self):
        benchmark = DIODEBenchmark()
        for name in ['DIODE', 'diode', 'ours', 'exact_linear_ode',
                     'exactlinearode', 'lbfgs', 'ours_lbfgs']:
            with self.subTest(name=name):
                self.assertEqual(benchmark._resolve_solvers(name), ['diode'])


if __name__ == '__main__':
    unittest.main()
