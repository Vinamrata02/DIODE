"""Exercise prediction from the last run using existing fitted artifacts."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
import yaml

from diode import DIODEBenchmark


runner = importlib.import_module("diode.runners.run_dataset")


class PredictionTests(unittest.TestCase):
    def test_predict_uses_last_run_datasets_and_supported_fits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            np.savetxt(root / "time.txt", [0.0, 2.0, 4.0])
            np.savetxt(root / "expression.txt", [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
            matrices = {"one": np.array([[0.0, 0.0], [2.0, 0.0]]), "two": np.eye(2)}
            for dataset, matrix in matrices.items():
                first_gene = "a" if dataset == "one" else "c"
                (root / f"{dataset}_genes.txt").write_text(f"{first_gene}\nb\n")
                config = {
                    "dataset": dataset, "expression": "expression.txt", "pseudotime": "time.txt",
                    "genes": f"{dataset}_genes.txt", "output_dir": f"outputs/{dataset}",
                }
                (root / f"configs/{dataset}.yaml").write_text(yaml.safe_dump(config))
                path = root / f"outputs/{dataset}/scode"
                path.mkdir(parents=True)
                (path / "scores_gene_order.csv").write_text(f"gene\n{first_gene}\nb\n")
                np.save(path / "scores_A.npy", matrix.T)

            def completed(config_path, *, solvers, **kwargs):
                return {"rows": [
                    {"dataset": config_path.stem, "solver": solver, "status": "ok"}
                    for solver in solvers
                ]}

            benchmark = DIODEBenchmark(repo_root=root)
            with self.assertRaisesRegex(ValueError, r"Call \.run"):
                benchmark.predict(["b"])
            with patch.object(runner, "run_dataset", side_effect=completed):
                benchmark.run(datasets=["two", "one"], solvers=["SCODE", "GENIE3"], progress=False)
                predictions = benchmark.predict(["b"])
                self.assertEqual(predictions["dataset"].unique().tolist(), ["two", "one"])
                self.assertEqual(predictions["solver"].unique().tolist(), ["SCODE"])
                self.assertEqual(predictions["gene"].unique().tolist(), ["b"])
                for dataset, frame in predictions.groupby("dataset"):
                    expected = [(expm(matrices[dataset] * t) @ [1, 2])[1] for t in (0, 0.5, 1)]
                    np.testing.assert_allclose(frame["predicted_expression"], expected, rtol=1e-6)
                    np.testing.assert_allclose(frame["observed_smooth"], [2, 3, 4])
                mixed = benchmark.predict(["a", "c"])
                self.assertEqual(set(zip(mixed["dataset"], mixed["gene"])), {("one", "a"), ("two", "c")})
                with self.assertRaisesRegex(ValueError, "two has no fitted expression for missing"):
                    benchmark.predict(["missing"])

                benchmark.run(datasets=["one"], solvers=["GENIE3"], progress=False)
                with self.assertRaisesRegex(ValueError, "DIODE, SCODE, or PHOENIX"):
                    benchmark.predict(["b"])
