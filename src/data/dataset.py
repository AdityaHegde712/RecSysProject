"""
Data loading utilities for HotelRec (recommendation task).

ItemKNN works directly with DataFrames — no PyTorch DataLoaders needed.
This module provides simple functions to load train/val/test splits
from parquet and get dataset dimensions.
"""

import os

import pandas as pd


def load_split(kcore_dir: str, split: str) -> pd.DataFrame:
    """Load a train/val/test split from parquet."""
    path = os.path.join(kcore_dir, f"{split}.parquet")
    return pd.read_parquet(path)


def get_n_users_items(kcore_dir: str) -> tuple[int, int]:
    """Read the full interaction file to get total user/item counts."""
    df = pd.read_parquet(os.path.join(kcore_dir, "interactions.parquet"))
    return df["user_id"].max() + 1, df["item_id"].max() + 1


def load_all_splits(kcore_dir: str) -> dict[str, pd.DataFrame]:
    """Load all three splits as a dict."""
    return {
        "train": load_split(kcore_dir, "train"),
        "val": load_split(kcore_dir, "val"),
        "test": load_split(kcore_dir, "test"),
    }


def get_user_positive_items(kcore_dir: str) -> dict[int, set[int]]:
    """
    Build a mapping of user_id -> set of all item_ids they interacted with
    across all splits. Used for negative sampling during evaluation.
    """
    splits = load_all_splits(kcore_dir)
    all_df = pd.concat(splits.values(), ignore_index=True)

    user_pos = {}
    for u, i in zip(all_df["user_id"].values, all_df["item_id"].values):
        if u not in user_pos:
            user_pos[u] = set()
        user_pos[u].add(i)

    return user_pos
