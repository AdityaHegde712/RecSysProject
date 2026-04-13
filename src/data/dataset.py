"""
Data loading for HotelRec.

Provides both simple DataFrame loaders (for ItemKNN) and PyTorch Dataset
classes (for neural models like GMF).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# DataFrame helpers (used by all models)
# ---------------------------------------------------------------------------

def load_split(kcore_dir: str, split: str) -> pd.DataFrame:
    """Load a train/val/test split from parquet."""
    return pd.read_parquet(os.path.join(kcore_dir, f"{split}.parquet"))


def get_n_users_items(kcore_dir: str) -> tuple[int, int]:
    """Read the full interaction file to get total user/item counts."""
    df = pd.read_parquet(os.path.join(kcore_dir, "interactions.parquet"))
    return df["user_id"].max() + 1, df["item_id"].max() + 1


def get_user_positive_items(kcore_dir: str) -> dict[int, set[int]]:
    """Build user_id -> set of all item_ids across all splits."""
    all_dfs = []
    for split in ("train", "val", "test"):
        all_dfs.append(load_split(kcore_dir, split))
    all_df = pd.concat(all_dfs, ignore_index=True)

    user_pos = {}
    for u, i in zip(all_df["user_id"].values, all_df["item_id"].values):
        if u not in user_pos:
            user_pos[u] = set()
        user_pos[u].add(i)
    return user_pos


# ---------------------------------------------------------------------------
# PyTorch Datasets (for neural models)
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """BPR training dataset with on-the-fly negative sampling."""

    def __init__(self, df: pd.DataFrame, n_items: int, num_negatives: int = 1):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.n_items = n_items
        self.num_negatives = num_negatives

        self.user_pos = {}
        for u, i in zip(self.users, self.items):
            if u not in self.user_pos:
                self.user_pos[u] = set()
            self.user_pos[u].add(i)

        self._len = len(self.users) * self.num_negatives

    def __len__(self):
        return self._len

    def _sample_neg(self, user: int) -> int:
        pos_set = self.user_pos.get(user, set())
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in pos_set:
                return neg

    def __getitem__(self, idx):
        orig_idx = idx // self.num_negatives
        user = self.users[orig_idx]
        pos_item = self.items[orig_idx]
        neg_item = self._sample_neg(user)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


class EvalInteractionDataset(Dataset):
    """Ranking eval: 1 positive + num_negatives random negatives per test row."""

    def __init__(self, df: pd.DataFrame, n_items: int,
                 user_pos_all: dict[int, set], num_negatives: int = 99,
                 seed: int = 42):
        self.n_items = n_items
        self.num_negatives = num_negatives

        rng = np.random.RandomState(seed)
        self.data = []

        for _, row in df.iterrows():
            u = int(row["user_id"])
            pos = int(row["item_id"])
            pos_set = user_pos_all.get(u, set())

            negs = []
            while len(negs) < num_negatives:
                j = rng.randint(0, n_items)
                if j not in pos_set and j != pos:
                    negs.append(j)

            self.data.append((u, pos, negs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, pos, negs = self.data[idx]
        items = [pos] + negs
        labels = [1] + [0] * len(negs)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    kcore_dir: str,
    batch_size: int = 256,
    num_negatives: int = 4,
    eval_negatives: int = 99,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders."""
    train_df = load_split(kcore_dir, "train")
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")

    n_users, n_items = get_n_users_items(kcore_dir)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    user_pos_all = {}
    for u, i in zip(all_df["user_id"].values, all_df["item_id"].values):
        if u not in user_pos_all:
            user_pos_all[u] = set()
        user_pos_all[u].add(i)

    train_ds = InteractionDataset(train_df, n_items, num_negatives)
    val_ds = EvalInteractionDataset(val_df, n_items, user_pos_all,
                                    eval_negatives, seed)
    test_ds = EvalInteractionDataset(test_df, n_items, user_pos_all,
                                     eval_negatives, seed)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
    }
