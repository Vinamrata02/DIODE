# GRNBoost2 (Arboreto)

GRNBoost2 is provided by the BSD-3-Clause licensed
[Arboreto package](https://github.com/aertslab/arboreto). This directory contains
the repository integration instructions and a standalone runner; Arboreto itself
is installed as a dependency rather than copied into this repository.

From the repository root, in the Python environment used by your notebook:

```bash
python -m pip install -e '.[grnboost2]'
python scripts/check_external_methods.py
```

The standard `environment.yml` and `requirements.txt` also include these Python
dependencies. To install them through the external-method setup helper, use
`DIODE_INSTALL_GRNBOOST2=1 bash scripts/setup_external_methods.sh` (this helper
also performs the existing setup for the other methods).

```python
from diode import DIODEBenchmark

results = DIODEBenchmark(dataset="data3").run(solvers=["GRNBoost2"])
```

All five dataset YAML files include `solvers.grnboost2`. The benchmark uses each
dataset's configured `baseline_expression`, shared with GENIE3 and the other
comparison methods: smoothed for Data1 and raw for the other four datasets,
unless the solver sets an explicit `expression` override. The gold standard is used only for evaluation.

The adapter follows `SCODE_Data3_human_OtherMethods.ipynb`: all input genes are
candidate regulators, seed 20, Arboreto's `SGBM_KWARGS`, and a one-worker,
one-thread in-process Dask cluster. It requests `include_meta=True` to avoid
Arboreto 0.1.6's empty-metadata-graph incompatibility with newer Dask. The default
configuration requires no scheduler service and works from notebooks.

Results are saved in `outputs/<dataset>/grnboost2/`:

- `GRNBoost2_edges.tsv`: `TF`, `target`, `importance`, sorted by importance.
- `A_GRNBoost2_target_by_regulator.txt`: target-by-regulator score matrix.
- `scores_A.csv`, `scores_A.npy`, `scores_gene_order.csv`: the standard benchmark outputs.

GRNBoost2 scores are unsigned feature importances, not signed regulatory effects.
Self-edges are zeroed and excluded by the benchmark's shared evaluation mask.

For user data without a reference network, run inference directly:

```bash
python external/GRNBoost2/run_grnboost2.py \
  --expression example_data/datamatrix.txt \
  --genes example_data/genes.txt \
  --output-dir outputs/example_data/grnboost2
```

The expression text file has **genes in rows and cells in columns**, matching the
repository's data files; the runner transposes it for Arboreto. Gene names are
one per line, in expression-row order. Python callers can instead pass a
cells-by-genes matrix to `diode.solvers.grnboost2.run_grnboost2_from_numpy`.
