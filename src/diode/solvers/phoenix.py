"""PHOENIX training, scoring, and prediction for genes-by-cells expression."""

from __future__ import annotations

import configparser
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


DEFAULT_MODEL_PARAMS = {
    "method": "dopri5",
    "neurons_per_layer": 200,
    "optimizer": "adam",
    "batch_size": 100,
    "init_lr": 1e-4,
    "weight_decay": 0.0,
    "val_split": 0.20,
    "noise": 0.0,
    "epochs": 100,
    "normalize_data": False,
    "scale_expression": 1.0,
}


def run_phoenix_from_numpy(
    X_sim,
    t_span,
    out_dir,
    *,
    model_params=None,
    phoenix_root=None,
    device="auto",
    seed=0,
    time_policy="rank_if_tied",
):
    """Train no-prior PHOENIX and return ``(A, CompletedProcess)``.

    ``X_sim`` has genes in rows and cells in columns. Cells are stably sorted
    by pseudotime. As in the five reproduction notebooks, tied times are
    replaced by evenly spaced rank time spanning the original range (or
    integer ranks if all times are equal). Set ``time_policy='error'`` to
    reject ties instead. The returned signed matrix has target genes in rows
    and source/regulator genes in columns.

    Parameters in ``model_params`` override the notebook defaults above.
    Training uses adjacent-cell transition MSE without a reference/prior
    network. The checkpoint with lowest epoch training MSE supplies the scores;
    ``val_split`` reserves transitions but does not select the checkpoint.
    Execution in a separate Python process isolates upstream module names and
    random seeds from other solvers. Checkpoints, history, inputs, and the
    raw/used time axes are saved in ``out_dir``.
    """
    X = np.asarray(X_sim, dtype=float)
    times = np.asarray(t_span, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 2:
        raise ValueError("PHOENIX expects a genes-by-cells matrix with at least two cells.")
    if times.size != X.shape[1]:
        raise ValueError("t_span length must equal the number of cells in X_sim.")
    if not np.isfinite(X).all() or not np.isfinite(times).all():
        raise ValueError("PHOENIX expression and pseudotime must be finite.")
    if time_policy not in {"rank_if_tied", "error"}:
        raise ValueError("time_policy must be 'rank_if_tied' or 'error'.")

    params = dict(DEFAULT_MODEL_PARAMS)
    overrides = dict(model_params or {})
    unknown = overrides.keys() - params.keys()
    if unknown:
        raise ValueError(f"Unknown PHOENIX model_params: {', '.join(sorted(unknown))}")
    params.update(overrides)
    for key in ("epochs", "batch_size", "neurons_per_layer"):
        value = params[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"PHOENIX {key} must be a positive integer.")
        params[key] = int(value)
    if params["optimizer"] not in {"adam", "rmsprop", "sgd", "adagrad"}:
        raise ValueError("PHOENIX optimizer must be adam, rmsprop, sgd, or adagrad.")
    if not 0 <= params["val_split"] < 1:
        raise ValueError("PHOENIX val_split must be in [0, 1).")
    for key in ("init_lr", "scale_expression"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"PHOENIX {key} must be finite and positive.")
    for key in ("noise", "weight_decay"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"PHOENIX {key} must be finite and nonnegative.")
    if not isinstance(params["normalize_data"], bool):
        raise ValueError("PHOENIX normalize_data must be a boolean.")

    code_dir = _phoenix_code_dir(phoenix_root)
    required = ("odenet.py", "datahandler.py", "csvreader.py", "figure_saver.py",
                "read_config.py", "torchdiffeq/__init__.py")
    missing = [name for name in required if not (code_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"PHOENIX runtime missing at {code_dir}: {', '.join(missing)}. "
            "Run scripts/setup_external_methods.sh or set phoenix_root."
        )

    order = np.argsort(times, kind="stable")
    X, raw_times = X[:, order], times[order]
    used_times = raw_times.copy()
    if np.any(np.diff(used_times) <= 0):
        if time_policy == "error":
            raise ValueError("PHOENIX requires strictly increasing pseudotime; tied times found.")
        lo, hi = float(raw_times.min()), float(raw_times.max())
        used_times = np.linspace(lo, hi, times.size) if hi > lo else np.arange(times.size, dtype=float)
    # Upstream CSV reader creates float32 times. Remove only an absolute offset
    # to avoid losing short intervals when pseudotime has a large origin.
    integration_times = (used_times - used_times[0]).astype(np.float32)
    if np.any(np.diff(integration_times) <= 0) or not np.isfinite(integration_times).all():
        raise ValueError("PHOENIX time intervals cannot be represented in float32.")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "input.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([X.shape[0], 1])
        writer.writerows(X)
        writer.writerow(integration_times)
    np.savetxt(out_dir / "raw_pseudotime.csv", raw_times, delimiter=",")
    np.savetxt(out_dir / "training_time.csv", used_times, delimiter=",")
    np.savetxt(out_dir / "integration_time.csv", integration_times, delimiter=",")
    np.savetxt(out_dir / "cell_order.csv", order, delimiter=",", fmt="%d")
    settings_path = out_dir / "run_settings.json"
    settings_path.write_text(json.dumps({
        "model_params": params, "device": str(device), "seed": int(seed),
        "time_policy": time_policy, "time_origin": float(used_times[0]),
    }, indent=2))

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(code_dir), str(out_dir)],
        capture_output=True, text=True, env=env,
    )
    (out_dir / "stdout.log").write_text(result.stdout)
    (out_dir / "stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"PHOENIX failed:\n{result.stdout}\n{result.stderr}")
    A = np.loadtxt(out_dir / "A.csv", delimiter=",", ndmin=2)
    if A.shape != (X.shape[0], X.shape[0]) or not np.isfinite(A).all():
        raise RuntimeError("PHOENIX returned an invalid score matrix.")
    return A, result


def _phoenix_code_dir(phoenix_root):
    if phoenix_root is None:
        phoenix_root = Path(__file__).resolve().parents[3] / "external" / "PHOENIX"
    return Path(phoenix_root).expanduser().resolve() / "ode_net" / "code"


def predict_phoenix_trajectory(fit_dir, phoenix_root, pseudotime, initial_state):
    """Integrate the best saved checkpoint and return expression in input units."""
    code_dir = _phoenix_code_dir(phoenix_root)
    for path in (code_dir / "odenet.py", fit_dir / "best_train_model.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_time = np.loadtxt(fit_dir / "raw_pseudotime.csv", delimiter=",", ndmin=1)
    if raw_time.shape != pseudotime.shape or not np.allclose(raw_time, pseudotime):
        raise ValueError(f"Saved PHOENIX pseudotime differs from the dataset in {fit_dir}.")

    with TemporaryDirectory(prefix="diode-phoenix-predict-") as work:
        input_path, output_path = Path(work) / "initial.npy", Path(work) / "prediction.npy"
        np.save(input_path, initial_state)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(code_dir), str(fit_dir), str(input_path), str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"PHOENIX prediction failed for {fit_dir}:\n{result.stderr or result.stdout}")
        return np.load(output_path)


def extract_phoenix_scores(model):
    """Official effects formula, transposed to target rows/source columns."""
    import torch

    with torch.no_grad():
        sums = model.net_sums.linear_out.weight
        prods = model.net_prods.linear_out.weight
        alpha = model.net_alpha_combine.linear_out.weight
        neurons = sums.shape[0]
        source_target = sums.T @ alpha[:, :neurons].T + prods.T @ alpha[:, neurons:].T
        source_target = source_target * torch.relu(model.gene_multipliers)
    return source_target.T.detach().cpu().numpy()


def _train(code_dir, out_dir):
    import torch

    # Upstream uses bare module imports and ships its own torchdiffeq. Keeping
    # these imports in a subprocess avoids polluting the notebook's sys.modules.
    sys.path.insert(0, str(code_dir))
    from datahandler import DataHandler
    from odenet import ODENet
    from read_config import read_arguments_from_file
    from torchdiffeq import odeint_adjoint as odeint

    run = json.loads((out_dir / "run_settings.json").read_text())
    params = run["model_params"]
    seed = run["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = run["device"]
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("PHOENIX supports cpu, cuda, or auto devices.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PHOENIX CUDA requested but CUDA is unavailable.")

    config = configparser.ConfigParser()
    config["settings"] = {key: str(value) for key, value in {
        **params, "cpu": device.type == "cpu", "viz": False,
        "batch_type": "single", "pretrained_model": False,
    }.items()}
    config_path = out_dir / "config.cfg"
    with config_path.open("w") as handle:
        config.write(handle)
    settings = read_arguments_from_file(str(config_path))
    # Normalization is handled here because upstream calls .numpy() on a
    # possibly CUDA tensor. Apply it before constructing the split tensors.
    handler = DataHandler.fromcsv(
        str(out_dir / "input.csv"), device, settings["val_split"],
        normalize=False, batch_type="single",
        batch_time=settings["batch_time"], batch_time_frac=settings["batch_time_frac"],
        noise=settings["noise"], img_save_dir=str(out_dir / "img"),
        scale_expression=settings["scale_expression"],
    )
    if settings["normalize_data"]:
        scale = max(float(x.abs().max().item()) for x in handler.data_pt)
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Cannot normalize PHOENIX data with zero/nonfinite maximum.")
        handler.data_pt = [x / scale for x in handler.data_pt]
        handler.data_np = [x / scale for x in handler.data_np]
        handler._create_validation_set_single()
    if not handler.train_data_length:
        raise ValueError("PHOENIX has no training transitions after the validation split.")
    model = ODENet(device, handler.dim, explicit_time=False, neurons=settings["neurons_per_layer"]).float()
    # ODENet.to() refers to nonexistent self.net; call the base implementation
    # to move all parameters, including gene_multipliers, onto the device.
    torch.nn.Module.to(model, device)
    optimizer_class = {
        "adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop,
        "sgd": torch.optim.SGD, "adagrad": torch.optim.Adagrad,
    }[settings["optimizer"]]
    optimizer = optimizer_class(model.parameters(), lr=settings["init_lr"], weight_decay=settings["weight_decay"])

    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, settings["epochs"] + 1):
        handler.reset_epoch()
        losses = []
        while not handler.epoch_done:
            batch, times, target = handler.get_batch(settings["batch_size"])
            optimizer.zero_grad()
            predictions = torch.stack([
                odeint(model, point, time, method=settings["method"])[1]
                for time, point in zip(times, batch)
            ])
            loss = torch.mean((predictions - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"PHOENIX training loss became nonfinite at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        history.append([epoch, train_loss])
        print(f"epoch={epoch:04d} train_mse={train_loss:.6e}", flush=True)
        if train_loss < best_loss:
            best_loss, best_epoch = train_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    torch.save(best_state, out_dir / "best_train_model.pt")
    model.load_state_dict(best_state)
    A = extract_phoenix_scores(model)
    if not np.isfinite(A).all():
        raise RuntimeError("PHOENIX extracted effects are nonfinite.")
    np.savetxt(out_dir / "A.csv", A, delimiter=",")
    np.savetxt(out_dir / "effects_source_target.csv", A.T, delimiter=",")
    np.savetxt(out_dir / "history.csv", np.asarray(history), delimiter=",",
               header="epoch,train_mse", comments="")
    (out_dir / "metadata.json").write_text(json.dumps({
        **run, "device_used": str(device), "best_epoch": best_epoch,
        "best_train_mse": best_loss, "matrix_convention": "target_rows_source_columns",
        "checkpoint_format": "torch_state_dict", "prior_used": False,
        "n_training_transitions": handler.train_data_length,
        "n_validation_transitions": handler.n_val,
    }, indent=2))


def _predict(code_dir, fit_dir, input_path, output_path):
    import torch

    sys.path.insert(0, str(code_dir))
    from odenet import ODENet
    from torchdiffeq import odeint

    torch.set_num_threads(1)
    run = json.loads((fit_dir / "run_settings.json").read_text())
    params = run["model_params"]
    times = np.loadtxt(fit_dir / "integration_time.csv", delimiter=",", ndmin=1)
    initial = np.load(input_path)
    expression_scale = params["scale_expression"]
    normalization = 1.0
    if params["normalize_data"]:
        fitted_input = np.loadtxt(fit_dir / "input.csv", delimiter=",", skiprows=1)[:-1]
        noise = np.random.RandomState(run["seed"]).normal(0, params["noise"], size=fitted_input.shape)
        normalization = np.abs(((fitted_input + noise) * expression_scale).astype(np.float32)).max()
        if not np.isfinite(normalization) or normalization == 0:
            raise ValueError("Cannot restore PHOENIX units with a zero or nonfinite normalization.")
    device = torch.device("cpu")
    model = ODENet(device, initial.size, explicit_time=False, neurons=params["neurons_per_layer"]).float()
    model.load_state_dict(torch.load(fit_dir / "best_train_model.pt", map_location=device, weights_only=True))
    model.eval()
    x0 = torch.as_tensor(initial * expression_scale / normalization, dtype=torch.float32)[None, :]
    t = torch.as_tensor(times, dtype=torch.float32)
    with torch.no_grad():
        fitted = odeint(model, x0, t, method=params["method"])[:, 0].numpy()
    np.save(output_path, fitted * normalization / expression_scale)


if __name__ == "__main__":
    args = [Path(arg) for arg in sys.argv[1:]]
    if len(args) == 2:
        _train(*args)
    else:
        _predict(*args)
