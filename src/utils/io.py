import os
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch: int, path: str):
    """Save model + optimizer state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model=None, optimizer=None):
    """
    Restore model state from a torch checkpoint.

    Pass the model (and optionally optimizer) to load weights into.
    Returns (model, epoch).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, ckpt["epoch"]
