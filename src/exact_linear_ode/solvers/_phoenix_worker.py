"""Isolated no-prior training loop shared by the PHOENIX adapter.

Matches the five PHOENIX reproduction notebooks: official DataHandler, ODENet,
adjoint integration, transition MSE, and best-training-loss score extraction.
"""

from __future__ import annotations

import configparser
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def extract_phoenix_scores(model):
    """Official effects formula, transposed to target rows/source columns."""
    with torch.no_grad():
        sums = model.net_sums.linear_out.weight
        prods = model.net_prods.linear_out.weight
        alpha = model.net_alpha_combine.linear_out.weight
        neurons = sums.shape[0]
        source_target = sums.T @ alpha[:, :neurons].T + prods.T @ alpha[:, neurons:].T
        source_target = source_target * torch.relu(model.gene_multipliers)
    return source_target.T.detach().cpu().numpy()


def train(code_dir, out_dir):
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


if __name__ == "__main__":
    train(Path(sys.argv[1]), Path(sys.argv[2]))
