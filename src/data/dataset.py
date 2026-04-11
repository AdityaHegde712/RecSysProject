"""
PyTorch Dataset classes for HotelRec (recommendation task only).

- InteractionDataset: returns (user_id, pos_item, neg_item) for BPR training
- EvalInteractionDataset: returns (user_id, item_list, labels) for ranking eval
- get_dataloaders(): builds train/val/test DataLoaders
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


class InteractionDataset(Dataset):
    """
    Dataset for implicit recommendation with negative sampling.
    Each __getitem__ returns (user, pos_item, neg_item).

    Negative items are sampled on-the-fly: for each positive interaction,
    we pick a random item the user hasn't interacted with.
    """

    def __init__(self, df: pd.DataFrame, n_items: int, num_negatives: int = 1):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.n_items = n_items
        self.num_negatives = num_negatives

        # build user -> set of positive items for fast negative sampling
        self.user_pos = {}
        for u, i in zip(self.users, self.items):
            if u not in self.user_pos:
                self.user_pos[u] = set()
            self.user_pos[u].add(i)

        # expand dataset: each positive gets num_negatives entries
        self._len = len(self.users) * self.num_negatives

    def __len__(self):
        return self._len

    def _sample_neg(self, user: int) -> int:
        """Sample a random item the user hasn't interacted with."""
        pos_set = self.user_pos.get(user, set())
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in pos_set:
                return neg

    def __getitem__(self, idx):
        # map back to the original positive pair
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
    """
    Evaluation dataset for ranking metrics.
    For each test interaction, pairs the positive item with num_negatives
    randomly sampled negatives. Returns (user, item_list, labels) where
    item_list[0] is the positive and the rest are negatives.
    """

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

            # sample negatives
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


def load_split(kcore_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(kcore_dir, f"{split}.parquet")
    return pd.read_parquet(path)


def get_n_users_items(kcore_dir: str) -> tuple[int, int]:
    """Read full interaction file to get total user/item counts."""
    df = pd.read_parquet(os.path.join(kcore_dir, "interactions.parquet"))
    return df["user_id"].max() + 1, df["item_id"].max() + 1


def get_dataloaders(
    kcore_dir: str,
    batch_size: int = 256,
    num_negatives: int = 4,
    eval_negatives: int = 99,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders for the recommendation task.

    Args:
        kcore_dir: path to the k-core processed directory
        batch_size: batch size for training
        num_negatives: negatives per positive (training)
        eval_negatives: negatives per positive (eval)
        num_workers: dataloader workers
        seed: random seed for eval negative sampling
    """
    train_df = load_split(kcore_dir, "train")
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")

    n_users, n_items = get_n_users_items(kcore_dir)

    # collect all positive interactions for negative sampling
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
