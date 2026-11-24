"""Verify smoothed baseline inputs and structural DIODE diagonal masking."""

import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from diode import DIODE, DIODEBenchmark
from diode.runners.run_dataset import run_dataset


ROOT = Path(__file__).resolve().parents[1]
runner = importlib.import_module("diode.runners.run_dataset")


def write_fixture(root):
    (root / "configs").mkdir()
    time = np.linspace(0, 1, 20)
    smooth = np.vstack([1 + time, 2 - time / 2, .5 + time ** 2])
    reference = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    np.savetxt(root / "smooth.txt", smooth)
    np.savetxt(root / "raw.txt", smooth + 100)
    np.savetxt(root / "time.txt", time)
    np.savetxt(root / "reference_network.csv", reference, delimiter=",")
    (root / "genes.txt").write_text("a\nb\nc\n")
    config = dict(
        dataset="tiny", expression="smooth.txt", baseline_expression="smooth.txt",
        pseudotime="time.txt", genes="genes.txt", reference_network_dir=".",
        output_dir="outputs/tiny",
        solvers={"diode": dict(
            seed=0, device="cpu", perturbation_print_every=0,
            cv=dict(n_splits=2, n_bins=2, seed=0),
            training=dict(mask_diagonal=True, epochs=2, lbfgs_max_iter=2,
                          lbfgs_line_search_fn=None, print_every=100, dtype="float64",
                          stop_on_loss=False, stop_on_parameter=False),
            hyperparameter_grid=dict(huber_delta=[.5], l2_lambda=[1e-5]),
        )},
    )
    return config, smooth


class TrainingPolicyTests(unittest.TestCase):
    def test_only_data1_matches_source_notebook_policy(self):
        benchmark = DIODEBenchmark(repo_root=ROOT)
        configs = [benchmark.config_path(dataset) for dataset in benchmark.datasets]
        self.assertEqual(len(configs), 5)
        for path in configs:
            config = yaml.safe_load(path.read_text())
            settings = config["solvers"]["diode"]
            self.assertNotIn("huber_delta", settings["training"])
            self.assertNotIn("l2_lambda", settings["training"])
            self.assertNotIn("tolerance_change", settings["hyperparameter_grid"])
            self.assertIsInstance(settings["training"]["tolerance_change"], (int, float))
            if config["dataset"] == "data1":
                self.assertEqual(config["baseline_expression"], config["expression"])
                self.assertFalse(settings["training"]["mask_diagonal"])
                self.assertEqual(settings["training"]["observation_dtype"], "float32")
                self.assertEqual(settings["training"]["patience_level"], 3)
                self.assertEqual(settings["training"]["optimizer_tolerance_grad"], 1e-3)
                self.assertEqual(settings["training"]["stop_tolerance_grad"], 1e-3)
                self.assertFalse(settings["cv"]["force_endpoints_in_training"])
                self.assertEqual(settings["evaluation"]["primary_score"], "absA")
            else:
                self.assertFalse(settings["training"].get("mask_diagonal", False))
                self.assertNotEqual(config["baseline_expression"], config["expression"])
            self.assertTrue((ROOT / config["baseline_expression"]).is_file())

    def test_all_baseline_dispatches_receive_smooth_data_and_ignore_diagonal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, smooth = write_fixture(root)
            (root / "configs/tiny.yaml").write_text(yaml.safe_dump(config))
            benchmark = DIODEBenchmark(repo_root=root)
            solvers = [s for s in benchmark.solvers if s != "DIODE"]
            seen = []
            def compute(solver, solver_cfg, cfg, repo_root, data):
                np.testing.assert_array_equal(data["expression_cells_by_genes"], smooth.T)
                seen.append(solver)
                # Positive reference self-edges and enormous predicted diagonals
                # must not participate in the AUROC comparison.
                return np.array([[1000., 3., 0.], [0., 1000., 2.], [1., 0., 1000.]])
            with patch.object(benchmark, "_compute_baseline_scores", side_effect=compute):
                results = benchmark.run(solvers=solvers, progress=False, skip_unavailable=False)
            self.assertEqual(len(seen), len(solvers))
            self.assertTrue(results["status"].eq("ok").all())
            np.testing.assert_array_equal(results["auroc"], np.ones(len(solvers)))

    def test_mask_applies_to_every_cv_model_and_final_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = write_fixture(root)
            path = root / "configs/tiny.yaml"
            path.write_text(yaml.safe_dump(config))
            masks = []
            original = runner.train_lti_lbfgs_with_validation
            def capture(model, *args, **kwargs):
                self.assertEqual(kwargs["optimizer_tolerance_change"], 1e-4)
                self.assertEqual(kwargs["stop_tolerance_change"], 1e-4)
                mask = model.mask.detach().cpu().numpy()
                self.assertFalse(np.diag(mask).any())
                self.assertEqual(int(mask.sum()), 6)
                np.testing.assert_array_equal(np.diag(model.A_eff.detach().cpu()), np.zeros(3))
                masks.append(mask.copy())
                result = original(model, *args, **kwargs)
                np.testing.assert_array_equal(np.diag(model.A_eff.detach().cpu()), np.zeros(3))
                return result
            with patch.object(runner, "train_lti_lbfgs_with_validation", side_effect=capture), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = run_dataset(path)
            self.assertEqual(len(masks), 3)  # Two CV folds and one final fit.
            output = root / "outputs/tiny"
            np.testing.assert_array_equal(np.diag(np.load(output / "final_refit_A.npy")), np.zeros(3))
            stored_mask = np.load(output / "training_network_mask.npy")
            np.testing.assert_array_equal(stored_mask, masks[0])
            self.assertTrue(result["metrics"]["mask_diagonal"])
            self.assertEqual(result["metrics"]["tolerance_change"], 1e-4)
            perturb = np.loadtxt(output / "perturb_delta_val_loss_scores.csv", delimiter=",")
            np.testing.assert_array_equal(np.diag(perturb), np.zeros(3))

    def test_fit_without_reference_and_evaluate_after_adding_it(self):
        for reference_dir in (None, "."):
            with self.subTest(reference_dir=reference_dir), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, _ = write_fixture(root)
                reference_path = root / "reference_network.csv"
                reference_data = reference_path.read_bytes()
                reference_path.unlink()
                if reference_dir is None:
                    config.pop("reference_network_dir")
                path = root / "configs/tiny.yaml"
                path.write_text(yaml.safe_dump(config))
                model = DIODE(path)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = model.fit()
                self.assertNotIn("auroc_perturb", result["metrics"])
                self.assertTrue((model.output_dir_ / "final_refit_A.npy").is_file())
                self.assertTrue((model.output_dir_ / "perturb_delta_val_loss_scores.csv").is_file())
                with self.assertRaisesRegex(ValueError, "Provide a reference to run evaluation"):
                    model.evaluate()
                reference_path.write_bytes(reference_data)
                config["reference_network_dir"] = "."
                path.write_text(yaml.safe_dump(config))
                explicit = model.evaluate(np.loadtxt(reference_path, delimiter=","))
                self.assertEqual(model.evaluate()["auroc_perturb"], explicit["auroc_perturb"])

    def test_default_and_explicit_mask_setting(self):
        for configured, expected_diagonal in ((None, True), (False, True), (True, False)):
            with self.subTest(mask_diagonal=configured), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, _ = write_fixture(root)
                if configured is None:
                    config["solvers"]["diode"]["training"].pop("mask_diagonal")
                else:
                    config["solvers"]["diode"]["training"]["mask_diagonal"] = configured
                path = root / "configs/tiny.yaml"
                path.write_text(yaml.safe_dump(config))
                def capture(d, mask, *args):
                    np.testing.assert_array_equal(np.diag(mask.cpu().numpy()),
                                                  np.full(3, expected_diagonal))
                    raise RuntimeError("verified initialization mask")
                with patch.object(runner, "_new_model", side_effect=capture), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(RuntimeError, "verified initialization mask"):
                    run_dataset(path)

    def test_flat_config_requires_nested_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = write_fixture(root)
            config.update(config.pop("solvers")["diode"])
            path = root / "configs/tiny.yaml"
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, r"requires solvers\.diode\.hyperparameter_grid"):
                run_dataset(path)

    def test_masked_parameters_have_no_diagonal_gradient(self):
        from diode.solvers.diode_lbfgs import LTIExact_Steady
        mask = torch.ones((3, 3), dtype=torch.bool)
        mask.fill_diagonal_(False)
        model = LTIExact_Steady(3, mask, dtype=torch.float64)
        initial = model.A_eff.detach().clone()
        loss = (model.A_eff - torch.ones_like(model.A_eff)).square().sum()
        loss.backward()
        np.testing.assert_array_equal(np.diag(model.W.grad.numpy()), np.zeros(3))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        optimizer.step()
        np.testing.assert_array_equal(np.diag(model.A_eff.detach().numpy()), np.zeros(3))
        self.assertFalse(torch.equal(initial, model.A_eff))


if __name__ == "__main__":
    unittest.main()
