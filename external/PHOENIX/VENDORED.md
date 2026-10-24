# PHOENIX vendoring notes

Source: https://github.com/QuackenbushLab/phoenix
Revision: b18317663aada933fa26402df6aad86d042bd13b
License: MIT (see LICENSE.txt).

This directory includes the unmodified PHOENIX runtime modules and its bundled
torchdiffeq 0.1.1 implementation from ode_net/code, plus the official extraction
script as a formula reference. Unrelated datasets, trained models, bytecode,
plots, and hard-coded training scripts are omitted.

The package adapter in src/exact_linear_ode/solvers/phoenix.py and its worker
parameterize the no-prior training loop from the five dataset notebooks.
The worker uses the bundled integrator, so a separate torchdiffeq installation
is unnecessary. Runtime dependencies (NumPy, PyTorch, Matplotlib) are already
included in the main package/environment.

Compatibility fixes live in the worker: move all model parameters with
torch.nn.Module.to, normalize without calling numpy on CUDA tensors, and save
state dictionaries instead of pickled modules or splitting dotted paths.
The upstream modules are unchanged. Scores use the best epoch training loss,
matching the notebooks; the reserved validation transitions do not select it.
