"""
Sequential dataset utilities for SASRec / GRU4Rec / BERT4Rec-style models.

Builds chronologically ordered per-user item sequences from the shared
20-core splits. All downstream sequence models consume the same loader
so their numbers are directly comparable.

The ``date`` column in HotelRec has ~month-level granularity, so ties
within a user are broken by original DataFrame order (stable sort).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_user_sequences(
    kcore_dir: str,
    max_seqlen: int = 50,
) -> tuple[dict[int, list[int]], int, int]:
    """Load train split and build user_id -> [item_1, item_2, ...] sorted
    chronologically. Truncates each sequence to the *last* ``max_seqlen``
    items (keeps recency, drops old history).

    Returns (sequences, n_users, n_items).
    """
    train_path = os.path.join(kcore_dir, "train.parquet")
    df = pd.read_parquet(train_path)
    df = df.sort_values(["user_id", "date"], kind="stable")

    seqs: dict[int, list[int]] = {}
    for u, sub in df.groupby("user_id", sort=False):
        items = sub["item_id"].astype(np.int64).tolist()
        if len(items) > max_seqlen:
            items = items[-max_seqlen:]
        seqs[int(u)] = items

    n_users = int(df["user_id"].max()) + 1
    n_items = int(
        pd.read_parquet(os.path.join(kcore_dir, "interactions.parquet"))[
            "item_id"
        ].max()
    ) + 1
    return seqs, n_users, n_items


# ---------------------------------------------------------------------------
# Training dataset (next-item prediction)
# ---------------------------------------------------------------------------

class NextItemDataset(Dataset):
    """Produces (input_sequence, target_item, negative_item) triples.

    Each user contributes ``len(seq) - 1`` samples: positions 1..L-1 are
    targets, and the corresponding prefix [0..t-1] is the input.
    Sequences shorter than ``max_seqlen`` are left-padded with pad_id = 0;
    item ids in the data are shifted +1 so that 0 is free for padding.

    Negative sampling is uniform over items (0..n_items-1) on the caller
    side, re-shifted by +1 for pad consistency.
    """

    def __init__(
        self,
        sequences: dict[int, list[int]],
        n_items: int,
        max_seqlen: int = 50,
        user_positives: dict[int, set[int]] | None = None,
        num_negatives: int = 1,
        seed: int = 42,
    ):
        self.max_seqlen = max_seqlen
        self.n_items = n_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        # Flatten: one sample per (user, target_position).
        self.user_ids: list[int] = []
        self.target_positions: list[int] = []
        self.sequences = sequences
        for u, seq in sequences.items():
            # t = 1..len(seq)-1  (need at least 1-item history)
            for t in range(1, len(seq)):
                self.user_ids.append(u)
                self.target_positions.append(t)

        # User positive-set for negative sampling (avoid positives).
        self.user_positives = user_positives or {
            u: set(seq) for u, seq in sequences.items()
        }

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx):
        u = self.user_ids[idx]
        t = self.target_positions[idx]
        full = self.sequences[u]

        # History up to but not including target.
        hist = full[:t]
        target = full[t]

        # Left-pad to max_seqlen with 0 (pad id), shift item ids by +1.
        hist_shifted = [i + 1 for i in hist[-self.max_seqlen:]]
        padded = [0] * (self.max_seqlen - len(hist_shifted)) + hist_shifted
        target_shifted = target + 1

        # Negative samples.
        pos_set = self.user_positives.get(u, set())
        negs = []
        while len(negs) < self.num_negatives:
            c = int(self.rng.integers(0, self.n_items))
            if c not in pos_set:
                negs.append(c + 1)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target_shifted, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Eval dataset (1-vs-99, but with sequence input)
# ---------------------------------------------------------------------------

class SequentialEvalDataset(Dataset):
    """For each (user, held-out-item) in val/test, produce (user_sequence,
    100-item candidate tensor) with the positive item at index 0.

    The user's sequence is the full training history (truncated to
    max_seqlen). Items are +1-shifted to match the training dataset's
    pad convention.
    """

    def __init__(
        self,
        sequences: dict[int, list[int]],
        eval_df: pd.DataFrame,
        n_items: int,
        user_positives: dict[int, set[int]],
        max_seqlen: int = 50,
        n_negatives: int = 99,
        seed: int = 42,
    ):
        self.sequences = sequences
        self.n_items = n_items
        self.max_seqlen = max_seqlen
        self.user_positives = user_positives
        self.rng = np.random.default_rng(seed)

        # Flatten eval pairs; dedupe (user, item) to match rank eval.
        pairs = eval_df[["user_id", "item_id"]].astype(np.int64).values
        self.pairs = pairs

        # Pre-sample 99 negatives per row deterministically for consistency.
        self.negatives = []
        for u, pos in pairs:
            pos_set = user_positives.get(int(u), set())
            negs = []
            while len(negs) < n_negatives:
                c = int(self.rng.integers(0, n_items))
                if c not in pos_set and c != int(pos):
                    negs.append(c)
            self.negatives.append(negs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos = int(self.pairs[idx][0]), int(self.pairs[idx][1])
        hist = self.sequences.get(u, [])
        hist_shifted = [i + 1 for i in hist[-self.max_seqlen:]]
        padded = [0] * (self.max_seqlen - len(hist_shifted)) + hist_shifted

        cands = [pos] + self.negatives[idx]      # raw ids
        cands_shifted = [c + 1 for c in cands]    # for model scoring

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(cands_shifted, dtype=torch.long),
            torch.tensor(cands, dtype=torch.long),   # raw ids for metric accounting
        )


# ---------------------------------------------------------------------------
# User positive helpers (shared across splits)
# ---------------------------------------------------------------------------

def build_user_positives(kcore_dir: str) -> dict[int, set[int]]:
    """Union of (user, item) pairs across all splits. Used by negative
    sampling to exclude known positives from the candidate pool."""
    pos: dict[int, set[int]] = {}
    for split in ("train", "val", "test"):
        df = pd.read_parquet(os.path.join(kcore_dir, f"{split}.parquet"))
        for u, i in zip(df["user_id"].values, df["item_id"].values):
            pos.setdefault(int(u), set()).add(int(i))
    return pos


# ---------------------------------------------------------------------------
# DataLoader factory for the bakeoff
# ---------------------------------------------------------------------------

def get_sequential_dataloaders(
    kcore_dir: str,
    batch_size: int = 256,
    max_seqlen: int = 50,
    num_negatives: int = 1,
    eval_negatives: int = 99,
    seed: int = 42,
):
    seqs, n_users, n_items = build_user_sequences(kcore_dir, max_seqlen)
    user_pos = build_user_positives(kcore_dir)

    train_ds = NextItemDataset(
        seqs, n_items, max_seqlen=max_seqlen,
        user_positives=user_pos, num_negatives=num_negatives, seed=seed,
    )
    val_df  = pd.read_parquet(os.path.join(kcore_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(kcore_dir, "test.parquet"))

    val_ds = SequentialEvalDataset(
        seqs, val_df, n_items, user_pos,
        max_seqlen=max_seqlen, n_negatives=eval_negatives, seed=seed + 1,
    )
    test_ds = SequentialEvalDataset(
        seqs, test_df, n_items, user_pos,
        max_seqlen=max_seqlen, n_negatives=eval_negatives, seed=seed + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "n_users": n_users,
        "n_items": n_items,
        "sequences": seqs,
        "user_positives": user_pos,
    }
