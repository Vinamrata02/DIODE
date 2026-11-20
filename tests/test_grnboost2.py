"""GRNBoost2 orientation, lifecycle, benchmark, and optional real-backend tests."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

from diode import DIODEBenchmark
from diode.solvers.grnboost2 import run_grnboost2_from_numpy


REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_backend(links):
    core = types.ModuleType("arboreto.core")
    core.SGBM_KWARGS = dict(n_estimators=5000, learning_rate=.01, max_features=.1, subsample=.9)
    core.EARLY_STOP_WINDOW_LENGTH = 25
    core.create_graph = MagicMock(return_value=("links_graph", "unused_metadata_graph"))
    distributed = types.ModuleType("distributed")
    cluster = distributed.LocalCluster = MagicMock()
    client = distributed.Client = MagicMock()
    client.return_value.__enter__.return_value.compute.return_value = links
    return {"arboreto": types.ModuleType("arboreto"), "arboreto.core": core,
            "distributed": distributed}, core, cluster, client


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GRNBoost2Tests(unittest.TestCase):
    def test_orientation_gene_order_missing_edges_and_cluster_cleanup(self):
        # A nonalphabetic order and asymmetric edges expose a transpose or reindex bug.
        genes = ["z", "a", "m"]
        links = pd.DataFrame({"TF": ["z", "a", "z"], "target": ["m", "z", "z"],
                              "importance": [3., 7., 99.]})
        modules, core, cluster, client = fake_backend(links)
        original_kwargs = dict(core.SGBM_KWARGS)
        expected = np.array([[0., 7., 0.], [0., 0., 0.], [3., 0., 0.]])
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            X = np.arange(30).reshape(10, 3)
            A = run_grnboost2_from_numpy(X, genes, output_dir=directory,
                                        regressor_kwargs={"n_estimators": 30})
            np.testing.assert_array_equal(A, expected)
            np.testing.assert_array_equal(np.loadtxt(Path(directory) / "A_GRNBoost2_target_by_regulator.txt"), A)
            saved = pd.read_csv(Path(directory) / "GRNBoost2_edges.tsv", sep="\t")
            self.assertEqual(saved["importance"].tolist(), [7., 3.])
            self.assertFalse((saved["TF"] == saved["target"]).any())
        graph_kwargs = core.create_graph.call_args.kwargs
        np.testing.assert_array_equal(graph_kwargs["expression_matrix"], X)
        self.assertEqual(graph_kwargs["gene_names"], genes)
        self.assertEqual(graph_kwargs["tf_names"], genes)
        self.assertTrue(graph_kwargs["include_meta"])
        self.assertEqual(graph_kwargs["seed"], 20)
        self.assertEqual(graph_kwargs["regressor_type"], "GBM")
        self.assertEqual(graph_kwargs["regressor_kwargs"]["n_estimators"], 30)
        self.assertEqual(graph_kwargs["early_stop_window_length"], 25)
        self.assertEqual(core.SGBM_KWARGS, original_kwargs)
        cluster.assert_called_once_with(n_workers=1, threads_per_worker=1,
                                        processes=False, dashboard_address=None)
        client.return_value.__enter__.return_value.compute.assert_called_once_with("links_graph", sync=True)
        cluster.return_value.__exit__.assert_called_once()
        client.return_value.__exit__.assert_called_once()

    def test_backend_failure_closes_owned_cluster(self):
        modules, _, cluster, client = fake_backend(pd.DataFrame())
        client.return_value.__enter__.return_value.compute.side_effect = RuntimeError("training failed")
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "training failed"):
            run_grnboost2_from_numpy([[1., 2.], [3., 4.]], ["a", "b"])
        cluster.return_value.__exit__.assert_called_once()
        client.return_value.__exit__.assert_called_once()

    def test_empty_links_preserve_square_zero_matrix(self):
        links = pd.DataFrame({"TF": pd.Series(dtype=str), "target": pd.Series(dtype=str),
                              "importance": pd.Series(dtype=float)})
        modules, _, _, _ = fake_backend(links)
        with patch.dict(sys.modules, modules):
            A = run_grnboost2_from_numpy(pd.DataFrame([[1., 2.], [3., 4.]], columns=["x", "y"]))
        np.testing.assert_array_equal(A, np.zeros((2, 2)))

    def test_invalid_inputs_fail_before_importing_optional_packages(self):
        cases = [([1., 2.], None, {}), ([[1., 2.]], None, {}),
                 ([[1., np.nan], [2., 3.]], None, {}),
                 ([[1., 2.], [3., 4.]], ["a", "a"], {}),
                 ([[1., 2.], [3., 4.]], ["a"], {}),
                 ([[1., 2.], [3., 4.]], ["a", " "], {}),
                 ([[1., 2.], [3., 4.]], None, {"n_workers": 0}),
                 ([[1., 2.], [3., 4.]], None, {"threads_per_worker": 0}),
                 ([[1., 2.], [3., 4.]], None, {"seed": -1}),
                 ([[1., 2.], [3., 4.]], None, {"early_stop_window_length": 0})]
        for X, genes, kwargs in cases:
            with self.subTest(kwargs=kwargs, genes=genes), self.assertRaises(ValueError):
                run_grnboost2_from_numpy(X, genes, **kwargs)

    def test_missing_backend_has_installation_instructions(self):
        with patch.dict(sys.modules, {"arboreto": None, "arboreto.core": None}):
            with self.assertRaisesRegex(ImportError, r"pip install -e.*grnboost2"):
                run_grnboost2_from_numpy([[1., 2.], [3., 4.]])

    def test_all_configs_and_public_solver_names(self):
        benchmark = DIODEBenchmark(repo_root=REPO_ROOT)
        self.assertIn("GRNBoost2", benchmark.solvers)
        for alias in ("GRNBoost2", "grnboost2", "grn_boost2"):
            self.assertEqual(benchmark._resolve_solvers(alias), ["grnboost2"])
        for path in (REPO_ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config["solvers"]["grnboost2"],
                             {"seed": 20, "n_workers": 1, "threads_per_worker": 1})

    def test_benchmark_loads_raw_expression_and_saves_shared_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
            A = np.array([[0., 2., 0.], [0., 0., 3.], [1., 0., 0.]])
            np.savetxt(root / "raw.txt", X)
            np.savetxt(root / "times.txt", [2., 1., 0.])
            (root / "genes.txt").write_text("a\nb\nc\n")
            np.savetxt(root / "reference_network.csv", (A != 0).astype(int), delimiter=",")
            config = dict(dataset="tiny", expression="unused_smooth.txt", baseline_expression="raw.txt",
                          pseudotime="times.txt", genes="genes.txt", reference_network_dir=".",
                          output_dir="outputs/tiny", solvers={"grnboost2": {"seed": 20, "n_workers": 1}})
            (root / "configs/tiny.yaml").write_text(yaml.safe_dump(config))
            with patch("diode.solvers.grnboost2.run_grnboost2_from_numpy", return_value=A) as run:
                results = DIODEBenchmark(repo_root=root).run(
                    solvers=["GRNBoost2"], skip_unavailable=False, progress=False)
            np.testing.assert_array_equal(run.call_args.args[0], X[:, ::-1].T)
            self.assertEqual(run.call_args.kwargs["gene_names"], ["a", "b", "c"])
            self.assertEqual(run.call_args.kwargs["seed"], 20)
            self.assertEqual(run.call_args.kwargs["output_dir"], root / "outputs/tiny/grnboost2")
            self.assertEqual(results.loc[0, "solver"], "GRNBoost2")
            self.assertEqual(results.loc[0, "status"], "ok")
            self.assertEqual(results.loc[0, "auroc"], 1.)
            np.testing.assert_array_equal(np.loadtxt(root / "outputs/tiny/grnboost2/scores_A.csv", delimiter=","), A)

    def test_standalone_runner_needs_no_reference_network(self):
        runner = load_script("grnboost2_standalone", REPO_ROOT / "external/GRNBoost2/run_grnboost2.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "expression.txt", [[1., 2., 3.], [4., 5., 6.]])
            (root / "genes.txt").write_text("x\ny\n")
            with patch.object(runner, "run_grnboost2_from_numpy", return_value=np.array([[0., 2.], [3., 0.]])) as run:
                runner.main(["--expression", str(root / "expression.txt"), "--genes", str(root / "genes.txt"),
                             "--output-dir", str(root / "outputs")])
            self.assertEqual(run.call_args.args[0].shape, (3, 2))
            self.assertEqual(run.call_args.kwargs["gene_names"], ["x", "y"])
            self.assertTrue((root / "outputs/scores_A.npy").exists())

    def test_preflight_lists_grnboost2(self):
        checker = load_script("grnboost2_preflight", REPO_ROOT / "scripts/check_external_methods.py")
        with patch.object(checker, "check_r_packages", return_value=(False, "unavailable")), \
             patch.object(checker, "check_matlab_engine", return_value=(False, "unavailable")):
            rows = checker.method_rows()
        self.assertEqual(len([row for row in rows if row[0] == "grnboost2"]), 1)

    @unittest.skipUnless(importlib.util.find_spec("arboreto") and importlib.util.find_spec("distributed"),
                         "Install .[grnboost2] to run the real Arboreto smoke test")
    def test_real_arboreto_fit_and_saved_orientation(self):
        rng = np.random.default_rng(20)
        X = rng.normal(size=(60, 4))
        X[:, 1] = 2 * X[:, 0] + rng.normal(scale=.1, size=60)
        names = ["d", "a", "c", "b"]
        with tempfile.TemporaryDirectory() as directory:
            A = run_grnboost2_from_numpy(X, names, output_dir=directory,
                                        regressor_kwargs={"n_estimators": 30, "max_features": 1.0})
            self.assertEqual(A.shape, (4, 4))
            self.assertTrue(np.isfinite(A).all())
            self.assertGreater(np.max(A), 0.)
            np.testing.assert_array_equal(np.diag(A), np.zeros(4))
            links = pd.read_csv(Path(directory) / "GRNBoost2_edges.tsv", sep="\t")
            reconstructed = links.pivot(index="target", columns="TF", values="importance").reindex(
                index=names, columns=names).fillna(0).to_numpy()
            np.testing.assert_allclose(A, reconstructed)


if __name__ == "__main__":
    unittest.main()
