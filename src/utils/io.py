import os
import pickle
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_model(model, path: str):
    """Save a non-neural model via pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str):
    """Load a non-neural model from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
