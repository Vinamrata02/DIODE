# DIODE

This repository contains code, data, notebooks, and reference networks for reproducing the DIODE experiments.

## Repository Layout

```text
DIODE/
├── src/diode/ # Installable package code
├── data/                 # Dataset inputs and reference networks
├── notebooks/            # Clean reproduction notebooks
└── configs/              # Dataset run settings
```

## Installation

Create the Conda environment, activate it, and install this repository in editable mode:

```bash
conda env create -f environment.yml
conda activate diode
pip install -e .
```

After installation, the package can be imported as `diode` and the command-line runners `diode-run` and `diode-run-all` are available.

The notebooks assume this editable install step has been completed.

## Setup Instructions

1. Create and activate the Python environment:

```bash
conda env create -f environment.yml
conda activate diode
pip install -e .
```

2. Make R and MATLAB available before starting Python or Jupyter. On a
module-based cluster, load the appropriate R and MATLAB modules for your
system. On a local machine, make sure `Rscript` and `matlab` resolve on `PATH`:

```bash
which Rscript
which matlab
```

3. Install the MATLAB Engine package into the same Python environment used by
the notebook. Use the MathWorks-supported local install:

```bash
cd "$(matlab -batch "disp(matlabroot)" | tail -1)/extern/engines/python"
python -m pip install .
```

or install the compatible PyPI package for your MATLAB release:

```bash
python -m pip install "matlabengine==<compatible-version>"
```

MathWorks installation guide:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

4. Install the R packages used by the R-backed methods:

```bash
Rscript -e 'install.packages(c("MASS", "infotheo", "pcalg", "msgps"), repos="https://cloud.r-project.org")'
```

If `pcalg` is unavailable from CRAN for your R version, install its
Bioconductor dependencies first:

```bash
Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install(c("graph", "RBGL")); install.packages("pcalg", repos="https://cloud.r-project.org")'
```

5. Check the external-method setup:

```bash
python scripts/check_external_methods.py
```

## Optional External Methods

The DIODE model runs from this repository. External comparison methods
used for reproduction are distributed under `external/` at the versions used by
the benchmark. The setup script can also refresh or restore those folders if
they are missing.

To check the external-method setup:

```bash
bash scripts/setup_external_methods.sh
python scripts/check_external_methods.py
```

SCODE requires `Rscript`, the R `MASS` package, and
`external/SCODE/SCODE.R`. The setup script checks MASS and installs it in
`.r-lib/scode` if needed. SCODE can use the Rscript saved by setup, an executable
on `PATH`, or one specified with `DIODE_RSCRIPT`. Dataset configs can also
specify a solver-specific `rscript_bin`.

GENIE3 and dynGENIE3 are distributed under `external/GENIE3` and
`external/dynGENIE3`. They run in Python and require the package dependencies
already listed in this environment, including `numpy`, `pandas`, and
`scikit-learn`.

PHOENIX is distributed under `external/PHOENIX` with its MIT license, source
revision, and bundled `torchdiffeq` integrator. It uses the existing NumPy,
PyTorch, and Matplotlib dependencies; no additional Python package, R, or MATLAB
installation is needed. The setup helper restores the pinned source if missing
and checks the active Python dependencies (`PHOENIX_PYTHON` can select the
interpreter). The preflight helper checks the vendored runtime as well.

Run PHOENIX through the same benchmark interface:

```python
from diode import DIODEBenchmark

results = DIODEBenchmark(dataset="data2").run(solvers=["phoenix"])
```

All five dataset configs reproduce the PHOENIX notebooks' no-prior workflow:
smoothed expression, 200 hidden neurons, Adam at `1e-4`, batch size 100,
100 epochs, and a 20% reserved transition split. The lowest epoch training MSE
selects the checkpoint, matching the notebooks; the reserved transitions are
not used for checkpoint selection or hyperparameter tuning. No reference
network is passed into PHOENIX training. Reference networks are used only by
the benchmark evaluation.

Each `solvers.phoenix.expression` selects the smoothed input independently of
the dataset-specific `baseline_expression` used by other methods. Data1 uses
smoothed baseline input; the other dataset configs retain raw baseline input. Cells are stably sorted
by pseudotime. Ties use evenly spaced rank time spanning the observed time
range (integer ranks if all times are equal), matching the original notebooks.
Set `time_policy: error` to reject tied times instead. The adapter records the
raw and training time axes, the offset-free float32 integration time, and cell
ordering in the output folder.

The low-level wrapper follows the SCODE calling convention:

```python
from diode.solvers.phoenix import run_phoenix_from_numpy

A, process = run_phoenix_from_numpy(
    X_genes_by_cells, pseudotime, "outputs/my_dataset/phoenix",
    device="auto", seed=0,
    model_params={"epochs": 100, "neurons_per_layer": 200},
)
```

`A` contains signed effects with target genes in rows and source/regulator genes
in columns. The adapter transposes PHOENIX's official extraction formula to
match the repository's reference networks. It saves `A.csv`, the original
`effects_source_target.csv`, `history.csv`, configuration, metadata, logs, and
best/final PyTorch state-dictionary checkpoints. Training runs in a separate
process using the active Python interpreter and the bundled integrator, keeping
PHOENIX's module imports and random seeds isolated from other methods. Use an
editable installation from a repository clone, as for the other vendored
methods, or supply `phoenix_root` pointing to a PHOENIX source checkout.

GRISLI requires MATLAB, MATLAB Engine for Python in the active Python
environment, and the bundled `external/GRISLI/GRISILI.m` adapter. On
module-based clusters, load MATLAB before starting Python or Jupyter so the
MATLAB command is available on `PATH`. Install the MATLAB Engine package into
the active Python environment using one of the MathWorks-supported methods:

```bash
cd "$(matlab -batch "disp(matlabroot)" | tail -1)/extern/engines/python"
python -m pip install .
```

or install the compatible PyPI package for your MATLAB release:

```bash
python -m pip install "matlabengine==<compatible-version>"
```

See the MathWorks MATLAB Engine installation guide for the current compatibility
table and virtual-environment notes:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

GRISLI also uses SPAMS MATLAB MEX files from
`external/GRISLI/spams-matlab-v2.6`. The upstream GRISLI copy includes compiled
MEX files for some platforms. If `scripts/check_external_methods.py` reports
that the platform-specific `mexLasso` file is missing, compile SPAMS with
MATLAB:

```bash
cd external/GRISLI/spams-matlab-v2.6
matlab -batch "compile; start_spams; assert(exist('mexLasso','file') > 0);"
```

The setup helper runs this same compilation step automatically when MATLAB is on
`PATH`. Set `GRISLI_COMPILE_SPAMS=0` to skip compilation:

```bash
GRISLI_COMPILE_SPAMS=0 bash scripts/setup_external_methods.sh
```

Keep the generated platform-specific MEX files under
`external/GRISLI/spams-matlab-v2.6/build/` for the platform used to run GRISLI.

PROB is distributed under `external/PROB` and uses MATLAB Engine for Python to
call `external/PROB/Executable functions/ODE_BayesianLasso_Python.m`.

SWING is distributed under `external/SWING`. It runs in Python and requires
`networkx`, `numpy`, `pandas`, `scipy`, and `scikit-learn`.

Linear regression, Lasso, MI, and IDA helper scripts are distributed under
`external/LR`, `external/Lasso`, `external/MI`, and `external/IDA`,
respectively. They require `Rscript` on `PATH`; MI/IDA additionally require the
R packages `infotheo` and `pcalg`, while Lasso requires `msgps`.

External solver paths and solver-specific parameters are configured in the
`solvers:` section of each dataset YAML under `configs/`. The paths are
repo-relative and point to the vendored method code under `external/`; edit those
sections only when a dataset needs solver-specific overrides such as SCODE
iterations, PHOENIX training parameters, or SWING window settings. A solver can
also set `expression` to override the shared `baseline_expression`.

## Notebooks

- `Reproduce_figures/Reproduce Fig.2d.ipynb`
- `Reproduce_figures/Reproduce Fig.2e.ipynb`
- `Example/data1_mouse_esc_pre.ipynb`
- `Example/data2_mouse_mef_in.ipynb`
- `Example/data3_human_esc_de.ipynb`
- `Example/04_data4_epiblast.ipynb`
- `Example/05_data4_trophectoderm.ipynb`

`Reproduce_figures/Reproduce Fig.2d.ipynb` (data1, Mouse MEF-PrE) and `Reproduce_figures/Reproduce Fig.2e.ipynb` (data3) demonstrate the package-level interface for running the solvers and reporting AUROC/AUPR metrics used to reproduce the paper results. They can be used to run the benchmark on one selected dataset or on multiple configured datasets. The example notebooks provide dataset-specific reproductions with the same installed package interface.

## Masked-entry imputation

`Reproduce_figures/Reproduce Fig.5e.ipynb` and `Reproduce_figures/Reproduce Fig.5f.ipynb`
run the five configured datasets with 2% and 5% masking, respectively. Each
notebook displays mean and sample standard deviation of RMSE across five masks
shared by DIODE, scNODE, MIOFlow, MAGIC, and ALRA.

Activate the environment used by your notebook before running the same
setup and checks as the AUROC methods.

```bash
conda activate ode-clone
bash scripts/setup_external_methods.sh
python scripts/check_external_methods.py
```

- Python dependencies are listed in `requirements.txt`. On Python 3.12 and
  later, the setup script installs MAGIC and scprep without resolving scprep's
  older pandas pin, then smoke-tests the imports in the active environment.
- The setup script fetches the reference checkouts into `external/scNODE`
  and `external/MIOFlow`. Use `DIODE_SCNODE_ROOT` and `DIODE_MIOFLOW_ROOT`
  to reuse checkouts elsewhere.
- The setup script installs the R package ALRA under `.r-lib/alra`. Set
  `DIODE_RSCRIPT` to an existing Rscript executable when it is outside your
  PATH. The repository remembers that executable. Once ALRA is available,
  the benchmark launches the R script from Python as part of each ALRA run.
  [ALRA source](https://github.com/KlugerLab/ALRA)

```python
from diode import DIODEBenchmark

results = DIODEBenchmark(dataset="data4_epiblast", masking=0.02).run(
    task="imputation", solvers=["DIODE", "MAGIC"], progress=True
)
```

Omit `dataset` to run all configured datasets. With `skip_unavailable=True`,
dependency errors are recorded in `results.attrs["failures"]`. The figure
notebooks require all five methods.

## Unmasked trajectory scores

`Reproduce_figures/Reproduce Fig.7.ipynb` uses `.run(task="imputation")`
with the masking fraction omitted. Explicit `masking=0.0` selects the
same workflow. Each method runs once. DIODE returns the fitted trajectory
from the full smooth expression matrix. The baselines use the raw
expression configured for this figure.

The table reports POS from Slingshot pseudotime, trajectory continuity,
and same-cell vertical cosine. Trajectory continuity is computed from the
1-nearest-neighbor fragmentation rate. The existing
`scripts/setup_external_methods.sh` installs [Slingshot](https://bioconductor.org/packages/slingshot/)
into `.r-lib/slingshot`, and `scripts/check_external_methods.py` checks it.

Data1–Data3 use the capture-time columns in the row order used by the
reference notebook. For Data1, the bundled labels concatenate the training
file followed by the test file. Epiblast and trophectoderm
use the supplied reference pseudotime for POS. The figure uses the smooth
reference matrix for epiblast vertical cosine and the reference notebook's
raw matrix for trophectoderm.

## Reference Networks

Reference networks live inside each `data/<dataset>/` folder. Every dataset has a canonical `reference_network.csv`; datasets with separate signed labels also include `reference_network_signed.csv`. In all cases, rows are target genes and columns are source/regulator genes.

## Run A Dataset

```bash
diode-run configs/data1.yaml
```

`run_dataset.py` mirrors the notebook workflow: it evaluates the configured hyperparameter grid across pseudotime-stratified CV folds, selects the best setting, refits that setting on the full ordered dataset, and reports AUROC/AUPR/signed-direction metrics from the final refit. The CV logs and selected hyperparameters are saved next to the final outputs.

Full runs can be expensive. For quick checks, lower `training.epochs`, `training.lbfgs_max_iter`, and the hyperparameter grid in a temporary copy of the dataset config.

## Run Your Own Dataset

To run DIODE on a new dataset, add one folder under `data/` and one matching YAML file under `configs/`. The benchmark interface discovers datasets from `configs/*.yaml`, so a new config with `dataset: my_dataset` will appear in:

```python
from diode import DIODEBenchmark

DIODEBenchmark.datasets
DIODEBenchmark.solvers
```

Recommended folder layout:

```text
data/my_dataset/
├── expression.txt
├── pseudotime.txt
├── genes.txt
├── reference_network.csv         # optional for DIODE fitting
└── reference_network_signed.csv  # optional
```

Input files:

- `expression.txt`: numeric expression matrix with genes as rows and cells/observations as columns. Whitespace-delimited text is supported.
- `pseudotime.txt`: one numeric pseudotime value per cell/observation. Its length must equal the number of columns in `expression.txt`.
- `genes.txt`: one gene name per row, in the same order as the rows of `expression.txt`.
- `reference_network.csv`: optional square reference network for DIODE fitting, required for AUROC/AUPR evaluation. Rows are target genes and columns are source/regulator genes, in the same gene order as `genes.txt`. Nonzero entries are treated as edges. The file can be a labeled comma-separated CSV with gene names as row/column labels, a headerless whitespace-delimited numeric matrix, or a headerless comma-separated numeric matrix.
- `reference_network_signed.csv`: optional square signed reference network with the same orientation, ordering, and accepted formats. Positive and negative values are used for signed-direction metrics.

Add a config such as:

```yaml
dataset: my_dataset
data_dir: data/my_dataset
reference_network_dir: data/my_dataset  # for evaluation
output_dir: outputs/my_dataset
expression: data/my_dataset/expression.txt
# Optional: use a different expression matrix for external solvers.
# When omitted, external solvers use `expression`.
baseline_expression: data/my_dataset/raw_expression.txt
pseudotime: data/my_dataset/pseudotime.txt
genes: data/my_dataset/genes.txt
solvers:
  diode:
    seed: 0
    device: auto
    training:
      loss_type: huber
      l1_lambda: 0.0
      lr: 1.0
      epochs: 1001
      print_every: 500
      lbfgs_max_iter: 20
      lbfgs_history_size: 10
      lbfgs_line_search_fn: strong_wolfe
      optimizer_tolerance_grad: 0.001
      stop_tolerance_grad: 0.001
      tolerance_change: 0.0001
      stop_on_loss: true
      stop_on_parameter: true
      stop_on_optimizer_step_size: false
      stop_on_grad: false
      patience_level: 5
      dtype: float64
      a_nnz_threshold: 0.000001
    # Tune only this grid. Keep the other DIODE settings at their defaults.
    hyperparameter_grid:
      huber_delta: [1.0]
      l2_lambda: [0.0000001]
```

Then run:

```bash
diode-run configs/my_dataset.yaml
```

or from Python:

```python
from diode import DIODEBenchmark

benchmark = DIODEBenchmark(dataset="my_dataset")
results = benchmark.run()
results
```

The runner sorts cells by pseudotime, performs the configured hyperparameter-selection workflow, refits the selected model on the full ordered dataset, and writes learned matrices to `output_dir`. It includes AUROC/AUPR when a reference network is available.

## Python Interface

```python
from diode import DIODEBenchmark

DIODEBenchmark.datasets
DIODEBenchmark.solvers

demo = DIODEBenchmark(dataset="data1")
results = demo.run()
results
```

Use `demo.run(solvers=["DIODE"])` to run only the DIODE solver.
External solver wrappers are exposed in `DIODEBenchmark.solvers`; if a
solver needs an unconfigured script, package, or precomputed score matrix, the
notebook interface reports it with `status == "not_available"` instead of
stopping the full run.

## CPU/GPU

Configs use `device: auto` by default. The runner uses CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU. Set `device: cpu` to force CPU, or `device: cuda` to require CUDA.

### Compatibility

The primary Python interface is `from diode import DIODE, DIODEBenchmark`. Reinstall the package with `python -m pip install -e . --no-deps --no-build-isolation` in your existing environment after updating this checkout. Saved notebook outputs are retained as historical run records.
