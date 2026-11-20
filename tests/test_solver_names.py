import unittest
from unittest.mock import patch

from diode import DIODEBenchmark
from diode.experiment import SolverUnavailableError


class SolverNameTests(unittest.TestCase):
    def test_public_names_resolve_to_existing_solver_ids(self):
        benchmark = DIODEBenchmark()
        expected = ['diode', 'genie3', 'grnboost2', 'dyn_genie3', 'scode', 'phoenix',
                    'grisli', 'prob', 'swing', 'lm', 'lasso', 'mi', 'ida_min', 'ida_mean']
        self.assertEqual(benchmark._resolve_solvers(benchmark.solvers), expected)
        self.assertEqual(benchmark._resolve_solvers(None), expected)
        self.assertEqual(benchmark._resolve_solvers('all'), expected)
        self.assertEqual(benchmark._resolve_solvers(expected), expected)

    def test_result_names_and_metrics(self):
        benchmark = DIODEBenchmark(dataset='data3')
        def run_solver(dataset, solver):
            return {'dataset': dataset, 'solver': solver, 'status': 'ok', 'auroc': 0.75}
        with patch.object(benchmark, '_run_solver', side_effect=run_solver) as run:
            results = benchmark.run(solvers=['DIODE', 'dynGENIE3', 'IDA_min', 'IDA_mean'], progress=False)
        self.assertEqual([call.args[1] for call in run.call_args_list],
                         ['diode', 'dyn_genie3', 'ida_min', 'ida_mean'])
        self.assertEqual(results['solver'].tolist(), ['DIODE', 'dynGENIE3', 'IDA', 'IDA'])
        self.assertEqual(results['auroc'].tolist(), [0.75] * 4)

    def test_unavailable_solver_has_public_name(self):
        benchmark = DIODEBenchmark(dataset='data3')
        with patch.object(benchmark, '_run_solver', side_effect=SolverUnavailableError('missing dependency')):
            results = benchmark.run(solvers=['PHOENIX'], progress=False)
        self.assertEqual(results.loc[0, 'solver'], 'PHOENIX')
        self.assertEqual(results.loc[0, 'status'], 'not_available')


if __name__ == '__main__':
    unittest.main()
