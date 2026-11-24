"""PHOENIX adapter regression tests; run with python -m unittest discover -s tests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from diode import DIODEBenchmark
from diode.solvers._phoenix_worker import extract_phoenix_scores
from diode.solvers.phoenix import predict_phoenix_trajectory, run_phoenix_from_numpy


REPO_ROOT = Path(__file__).resolve().parents[1]


class PhoenixTests(unittest.TestCase):
    def test_extraction_orientation_and_target_scaling(self):
        # An asymmetric case distinguishes a transpose or source/target scaling
        # error. This expected matrix follows the notebook's NumPy formula.
        sums = np.array([[1., 2.], [3., 4.]])
        prods = np.array([[5., 6.], [7., 8.]])
        alpha = np.array([[1., 0., 2., 0.], [0., 3., 0., 4.]])
        multipliers = np.array([[2., -1.]])
        def layer(weight):
            return SimpleNamespace(linear_out=SimpleNamespace(weight=torch.tensor(weight)))
        model = SimpleNamespace(
            net_sums=layer(sums), net_prods=layer(prods),
            net_alpha_combine=layer(alpha), gene_multipliers=torch.tensor(multipliers),
        )
        expected = ((sums.T @ alpha.T[:2] + prods.T @ alpha.T[2:]) * np.maximum(multipliers, 0)).T
        np.testing.assert_array_equal(extract_phoenix_scores(model), expected)
        np.testing.assert_array_equal(expected, [[22., 28.], [0., 0.]])

    def test_all_five_configs_use_notebook_expression_and_defaults(self):
        benchmark = DIODEBenchmark(repo_root=REPO_ROOT)
        self.assertEqual(benchmark._resolve_solvers("PHOENIX"), ["phoenix"])
        self.assertIn("PHOENIX", benchmark.solvers)
        paths = [benchmark.config_path(dataset) for dataset in benchmark.datasets]
        self.assertEqual(len(paths), 5)
        for path in paths:
            config = yaml.safe_load(path.read_text())
            settings = config["solvers"]["phoenix"]
            self.assertEqual(settings["expression"], config["expression"])
            self.assertTrue((REPO_ROOT / settings["expression"]).is_file())
            self.assertEqual(settings["model_params"]["epochs"], 100)
            self.assertEqual(settings["model_params"]["neurons_per_layer"], 200)

    def test_bad_inputs_fail_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [
                ([[1., 2.]], [0.], {}),
                ([[1., np.nan]], [0., 1.], {}),
                ([[1., 2.]], [0., 0.], {"time_policy": "error"}),
                ([[1., 2.]], [0., 1.], {"model_params": {"epochs": 0}}),
                ([[1., 2.]], [0., 1.], {"model_params": {"epoch": 1}}),
            ]
            for X, times, kwargs in cases:
                with self.subTest(kwargs=kwargs, times=times):
                    with self.assertRaises(ValueError):
                        run_phoenix_from_numpy(X, times, directory, **kwargs)

    def test_benchmark_uses_solver_expression_and_shared_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            A = np.array([[0., 2., 0.], [0., 0., -3.], [1., 0., 0.]])
            expression = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
            np.savetxt(root / "smooth.txt", expression)
            np.savetxt(root / "time.txt", [2., 1., 0.])
            (root / "genes.txt").write_text("a\nb\nc\n")
            np.savetxt(root / "reference_network.csv", (A != 0).astype(int), delimiter=",")
            np.savetxt(root / "reference_network_signed.csv", np.sign(A), delimiter=",")
            config = {
                "dataset": "tiny", "expression": "unused.txt", "baseline_expression": "unused_raw.txt",
                "pseudotime": "time.txt", "genes": "genes.txt", "reference_network_dir": ".",
                "output_dir": "outputs/tiny",
                "solvers": {"phoenix": {
                    "expression": "smooth.txt", "seed": 17, "device": "cpu",
                    "model_params": {"epochs": 1},
                }},
            }
            (root / "configs/tiny.yaml").write_text(yaml.safe_dump(config))
            with patch("diode.solvers.phoenix.run_phoenix_from_numpy", return_value=(A, None)) as run:
                result = DIODEBenchmark(repo_root=root).run(
                    solvers=["phoenix"], skip_unavailable=False, progress=False,
                )
            np.testing.assert_array_equal(run.call_args.args[0], expression[:, ::-1])
            np.testing.assert_array_equal(run.call_args.args[1], [0., 1., 2.])
            self.assertEqual(run.call_args.kwargs["seed"], 17)
            self.assertEqual(result.iloc[0]["status"], "ok")
            self.assertEqual(result.iloc[0]["auroc"], 1.)
            np.testing.assert_array_equal(np.loadtxt(root / "outputs/tiny/phoenix/scores_A.csv", delimiter=","), A)

    def test_real_training_tied_times_and_safe_checkpoint(self):
        # Dotted output paths used to break upstream ODENet.save(). Unsorted
        # tied times exercise stable cell ordering and rank-time conversion.
        with tempfile.TemporaryDirectory(prefix="phoenix.test.") as directory:
            X = np.array([[.8, 1., .9, .7], [.6, .5, .55, .65], [.6, .8, .7, .5]])
            A, process = run_phoenix_from_numpy(
                X, [.5, 0., 0., 1.], directory, device="cpu", seed=17,
                model_params={"epochs": 2, "neurons_per_layer": 10, "batch_size": 3,
                              "scale_expression": 2.0, "normalize_data": True},
            )
            self.assertEqual(process.returncode, 0)
            self.assertEqual(A.shape, (3, 3))
            self.assertTrue(np.isfinite(A).all())
            output = Path(directory)
            np.testing.assert_array_equal(np.loadtxt(output / "cell_order.csv"), [1, 2, 0, 3])
            np.testing.assert_allclose(np.loadtxt(output / "training_time.csv"), np.linspace(0, 1, 4))
            np.testing.assert_allclose(np.loadtxt(output / "effects_source_target.csv", delimiter=","), A.T)
            state = torch.load(output / "best_train_model.pt", map_location="cpu", weights_only=True)
            self.assertIn("gene_multipliers", state)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["prior_used"])
            self.assertIn(metadata["best_epoch"], [1, 2])
            self.assertEqual(np.loadtxt(output / "history.csv", delimiter=",", skiprows=1).shape, (2, 2))
            fitted = predict_phoenix_trajectory(
                output, REPO_ROOT / "external/PHOENIX", np.array([0.0, 0.0, 0.5, 1.0]), X[:, 1],
            )
            self.assertEqual(fitted.shape, X.T.shape)
            self.assertTrue(np.isfinite(fitted).all())
            np.testing.assert_allclose(fitted[0], X[:, 1])


if __name__ == "__main__":
    unittest.main()
