import contextlib
import importlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from tqdm import tqdm

import numpy as np

from diode import DIODEBenchmark
from diode.experiment import SolverUnavailableError
from diode.solvers import diode_lbfgs
from diode.utils.evaluations import (
    minmax_lower_is_better_score,
    ordered_fragmentation_k1,
    paper_pos_score,
    vertical_cosine_scores,
)

runner = importlib.import_module("diode.runners.run_dataset")
ROOT = Path(__file__).resolve().parents[1]


class ImputationTests(unittest.TestCase):
    def test_interpolation_preserves_observed_zeros_and_uses_both_fallbacks(self):
        times = np.array([3.0, 1.0, 0.0, 1.0, 1.0, 2.0, 4.0])
        observed = np.array([[6, 0], [0, 10], [0, 1], [0, 0], [0, 20], [0, 5], [0, 4]], dtype=float)
        mask = np.zeros_like(observed, dtype=bool)
        mask[[5, 6], 0] = True
        # The repeated times use the same argsort ordering as the source notebook.
        tied = np.argsort(times)[1:4]
        mask[tied[1], 1] = True
        observed[tied[1], 1] = 0
        ode = np.full_like(observed, 99)
        result = diode_lbfgs.impute_diode(observed, mask, times, ode)
        self.assertEqual(result[5, 0], 3)
        self.assertEqual(result[6, 0], 99)
        self.assertEqual(result[tied[1], 1], 99)
        np.testing.assert_array_equal(result[~mask], observed[~mask])

    def test_benchmarks_share_seeded_masks_and_compute_the_requested_table(self):
        benchmark = DIODEBenchmark(dataset="data1", repo_root=ROOT, masking=0.02)
        observed_by_method = {"magic": [], "alra": []}

        def record(method, observed):
            observed_by_method[method].append(observed.copy())
            return np.zeros_like(observed)

        with patch.object(runner, "_imputation_baselines", return_value={
            method: lambda observed, times, method=method: record(method, observed)
            for method in observed_by_method
        }), contextlib.redirect_stdout(io.StringIO()) as stdout:
            results = benchmark.run(solvers=["MAGIC", "ALRA"], progress=False)
        self.assertEqual(stdout.getvalue(), "Running the imputation benchmark with 2% masking.\n")
        self.assertEqual(list(results.columns), list(runner.IMPUTATION_COLUMNS))
        self.assertEqual(results["dataset"].tolist(), ["Mouse ESC-PrE"] * 2)
        self.assertEqual(results["method"].tolist(), ["MAGIC", "ALRA"])
        self.assertEqual(results["n_runs"].tolist(), [5, 5])

        truth = np.loadtxt(ROOT / "data/data1/datamatrix_ordered_smooth_5.txt").T
        run_rmses = []
        for seed in range(5):
            indices = np.random.default_rng(seed).choice(
                truth.size, size=round(truth.size * 0.02), replace=False,
            )
            expected = truth.ravel().copy()
            expected[indices] = 0
            for seen in observed_by_method.values():
                np.testing.assert_array_equal(seen[seed].ravel(), expected)
            run_rmses.append(np.sqrt(np.mean(truth.ravel()[indices] ** 2)))
        np.testing.assert_allclose(results["mean_RMSE"], np.mean(run_rmses))
        np.testing.assert_allclose(results["sample_SD_RMSE"], np.std(run_rmses, ddof=1))

    def test_notebook_progress_falls_back_to_text_when_widgets_are_unavailable(self):
        stdout = io.StringIO()
        benchmark = DIODEBenchmark(dataset="data1", repo_root=ROOT, masking=0.02)

        def predict(observed, times):
            self.assertIn("data1: run 1/5, MAGIC", stdout.getvalue())
            return observed

        with (
            patch("diode.experiment._running_in_notebook", return_value=True),
            patch("diode.experiment.IProgress", None),
            patch.object(runner, "_imputation_baselines", return_value={"magic": predict, "alra": predict}),
            contextlib.redirect_stdout(stdout),
        ):
            benchmark.run(task="imputation", solvers=["MAGIC", "ALRA"], progress=True)

        self.assertIn("data1: run 1/5, ALRA", stdout.getvalue())
        self.assertIn("1/10", stdout.getvalue())
        self.assertIn("10/10", stdout.getvalue())

    def test_imputation_and_auroc_use_the_same_nested_widget_bars(self):
        benchmark = DIODEBenchmark(dataset="data1", repo_root=ROOT)
        with (
            patch("diode.experiment._running_in_notebook", return_value=True),
            patch("diode.experiment.IProgress", object()),
            patch("diode.experiment.notebook_tqdm", side_effect=lambda *args, **kwargs: tqdm(*args, **kwargs, disable=True)) as create,
            patch.object(benchmark, "_run_solver", return_value={"dataset": "data1", "solver": "diode"}),
            patch.object(runner, "_imputation_baselines", return_value={"magic": lambda observed, times: observed}),
        ):
            benchmark.run(solvers="DIODE")
            DIODEBenchmark(dataset="data1", repo_root=ROOT, masking=0.02).run(solvers="MAGIC", task="imputation")

        self.assertEqual([call.kwargs["position"] for call in create.call_args_list], [0, 1, 0, 1])
        self.assertEqual([call.kwargs["total"] for call in create.call_args_list[1::2]], [1, 5])
        self.assertEqual(create.call_args_list[1].kwargs["bar_format"], create.call_args_list[3].kwargs["bar_format"])

    def test_task_and_solver_are_validated_before_dispatch(self):
        with patch.object(runner, "run_dataset") as run:
            with self.assertRaisesRegex(ValueError, "Choose from DIODE"):
                DIODEBenchmark(masking=0.05).run(solvers="GENIE3", progress=False)
            with self.assertRaisesRegex(ValueError, "masking must be"):
                DIODEBenchmark(masking=-0.02).run(task="imputation", progress=False)
            with self.assertRaisesRegex(ValueError, "requires task='imputation'"):
                DIODEBenchmark(masking=0.02).run(task="auroc", progress=False)
        run.assert_not_called()

    def test_unmasked_imputation_uses_raw_baselines_and_direct_diode_fit(self):
        raw = np.loadtxt(ROOT / "data/data1/datamatrix.txt").T
        smooth = np.loadtxt(ROOT / "data/data1/datamatrix_ordered_smooth_5.txt").T
        reference_time = np.loadtxt(ROOT / "data/data1/fig7_capture_time.txt")

        def baseline(observed, times):
            np.testing.assert_array_equal(observed, raw)
            return observed

        def fit(config_path, data, observed, order):
            np.testing.assert_array_equal(observed, smooth)
            return observed

        def infer(expression, *, repo_root, start_cell_index, runtime):
            self.assertEqual(start_cell_index, int(np.argmin(reference_time)))
            return -reference_time, "test"

        for masking in (None, 0.0):
            with (
                self.subTest(masking=masking),
                patch.object(runner, "_imputation_baselines", return_value={"magic": baseline}),
                patch.object(diode_lbfgs, "_fit_imputation_trajectory", side_effect=fit) as fit_call,
                patch.object(diode_lbfgs, "impute_diode") as interpolate,
                patch.object(runner, "ordered_fragmentation_k1", side_effect=[0.25, 0.75]),
                patch.object(runner, "_run_imputation") as masked_runner,
                patch("diode.utils.slingshot.resolve_slingshot_runtime"),
                patch("diode.utils.slingshot.infer_pseudotime_slingshot", side_effect=infer) as infer_call,
            ):
                result = DIODEBenchmark(dataset="data1", repo_root=ROOT, masking=masking).run(
                    task="imputation", solvers=["DIODE", "MAGIC"], progress=False,
                )
            fit_call.assert_called_once()
            interpolate.assert_not_called()
            masked_runner.assert_not_called()
            self.assertEqual(infer_call.call_count, 2)
            self.assertEqual(
                tuple(result.columns),
                ("dataset", "method", "pos_score", "trajectory_continuity_score", "vertical_cosine_score"),
            )
            np.testing.assert_allclose(result["pos_score"], 1.0)
            np.testing.assert_allclose(result["trajectory_continuity_score"], [1.0, 0.0])
            self.assertAlmostEqual(result.loc[1, "vertical_cosine_score"], 1.0 - 0.5 * np.all(raw == 0, axis=1).mean())

    def test_trajectory_score_formulas(self):
        np.testing.assert_allclose(
            minmax_lower_is_better_score([0.25, np.nan, 0.5, 0.75]),
            [1.0, np.nan, 0.5, 0.0],
        )
        np.testing.assert_allclose(minmax_lower_is_better_score([0.25, 0.25]), 1.0)
        self.assertEqual(paper_pos_score([0, 2, 1], [0, 1, 2]), 2 / 3)
        self.assertEqual(paper_pos_score([0, 0, 2], [0, 1, 2]), 1.0)
        self.assertTrue(np.isnan(paper_pos_score([1, 1, 1], [0, 1, 2])))
        self.assertAlmostEqual(vertical_cosine_scores([[1, 0], [1, 0]], [[-1, 0], [0, 0]])["vertical_cosine_score"], 0.25)
        self.assertEqual(ordered_fragmentation_k1([[0], [1], [10], [11]], [0, 1, 2, 3]), 1 / 3)

    def test_unavailable_dependency_uses_status_and_respects_strict_mode(self):
        def unavailable(observed, times):
            raise ImportError("missing package")

        benchmark = DIODEBenchmark(dataset="data1", repo_root=ROOT, masking=0.02)
        with patch.object(runner, "_imputation_baselines", return_value={"magic": unavailable}):
            result = benchmark.run(solvers="MAGIC", task="imputation", progress=False)
            with self.assertRaisesRegex(SolverUnavailableError, "missing package"):
                benchmark.run(solvers="MAGIC", task="imputation", progress=False, skip_unavailable=False)
        self.assertEqual(result.loc[0, "n_runs"], 0)
        self.assertTrue(np.isnan(result.loc[0, "mean_RMSE"]))
        self.assertIn("missing package", result.attrs["failures"][0]["message"])


if __name__ == "__main__":
    unittest.main()
