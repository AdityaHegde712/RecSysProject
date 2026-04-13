"""I/O utilities: config loading, model save/load (pickle + torch)."""

import os
import pickle
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -- Torch checkpoint save/load (for neural models) --

def save_checkpoint(model, optimizer, epoch: int, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, model=None, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, ckpt["epoch"]


# -- Pickle save/load (for non-neural models like ItemKNN) --

def save_model(model, path: str):
    """Save a non-neural model via pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str):
    """Load a non-neural model from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
