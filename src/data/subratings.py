"""
Sub-rating data loading for HotelRec.

HotelRec reviews include 6 aspect-level sub-ratings:
  Service, Cleanliness, Location, Value, Rooms, Sleep Quality

These are stored in the interactions parquet alongside the overall rating.
This module provides utilities to load and normalize sub-ratings for the
sub-rating decomposition model.

"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# the 6 sub-rating columns in HotelRec
SUBRATING_COLS = [
    "rating_service",
    "rating_cleanliness",
    "rating_location",
    "rating_value",
    "rating_rooms",
    "rating_sleep_quality",
]

# human-readable names
SUBRATING_NAMES = [
    "Service", "Cleanliness", "Location", "Value", "Rooms", "Sleep Quality",
]


def load_subratings(kcore_dir: str, split: str = "train") -> pd.DataFrame:
    """Load a split and extract sub-rating columns.

    Returns the DataFrame with user_id, item_id, rating, and sub-rating cols.
    Missing sub-ratings are filled with the overall rating as a fallback.
    """
    path = os.path.join(kcore_dir, f"{split}.parquet")
    df = pd.read_parquet(path)

    # check which sub-rating columns actually exist
    available = [c for c in SUBRATING_COLS if c in df.columns]
    missing = [c for c in SUBRATING_COLS if c not in df.columns]

    if missing:
        print(f"  Warning: missing sub-rating columns: {missing}")
        # fill missing sub-ratings with overall rating
        overall = df["rating"] if "rating" in df.columns else pd.Series(3.0, index=df.index)
        for col in missing:
            df[col] = overall

    return df


def get_subrating_stats(kcore_dir: str):
    """Print basic stats about sub-ratings in the dataset."""
    df = load_subratings(kcore_dir, "train")
    print(f"\nSub-rating statistics (train split, {len(df):,} rows):")
    for col, name in zip(SUBRATING_COLS, SUBRATING_NAMES):
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {name:15s}: mean={vals.mean():.2f}, "
                  f"std={vals.std():.2f}, "
                  f"missing={df[col].isna().sum()}")


class SubratingDataset(Dataset):
    """Training dataset with sub-ratings for the decomposition model.

    Returns (user, pos_item, neg_item, sub_ratings_tensor) where
    sub_ratings_tensor is shape (6,) with normalized sub-ratings.
    """

    def __init__(self, df: pd.DataFrame, n_items: int, num_negatives: int = 1):
        self.users = df["user_id"].values
        self.items = df["item_id"].values
        self.n_items = n_items
        self.num_negatives = num_negatives

        # extract sub-ratings as a (N, 6) array
        # normalize from [1, 5] to [0, 1]
        sub_cols = []
        for col in SUBRATING_COLS:
            if col in df.columns:
                vals = df[col].fillna(3.0).values.astype(np.float32)
            else:
                vals = np.full(len(df), 3.0, dtype=np.float32)
            sub_cols.append(vals)

        self.subratings = np.stack(sub_cols, axis=1)  # (N, 6)
        self.subratings = (self.subratings - 1.0) / 4.0  # normalize to [0, 1]

        # user positive set for negative sampling
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
        subs = self.subratings[orig_idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
            torch.from_numpy(subs),  # (6,) float32
        )
