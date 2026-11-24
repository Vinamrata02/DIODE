import unittest

from diode import DIODEBenchmark


class DIODECompatibilityTests(unittest.TestCase):
    def test_existing_solver_inputs_select_diode(self):
        benchmark = DIODEBenchmark()
        for name in ['DIODE', 'diode', 'ours', 'exact_linear_ode',
                     'exactlinearode', 'lbfgs', 'ours_lbfgs']:
            with self.subTest(name=name):
                self.assertEqual(benchmark._resolve_solvers(name), ['diode'])


if __name__ == '__main__':
    unittest.main()
