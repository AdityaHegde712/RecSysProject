"""
Popularity baseline: recommends items by global interaction frequency.

No personalization -- every user gets the same ranking.
Serves as a lower bound that any reasonable model should beat.
"""

import numpy as np
import pandas as pd
import torch


class PopularityBaseline:
    """Score items by how often they appear in the training set."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.item_scores = np.zeros(n_items, dtype=np.float32)

    def fit(self, train_df: pd.DataFrame):
        counts = train_df["item_id"].value_counts()
        for item_id, count in counts.items():
            self.item_scores[item_id] = count
        max_count = self.item_scores.max()
        if max_count > 0:
            self.item_scores = self.item_scores / max_count
        print(f"  PopularityBaseline: fit on {len(train_df):,} interactions, "
              f"{(self.item_scores > 0).sum():,} items with >0 interactions")
        return self

    def predict_batch(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Return popularity scores for a batch. Shape: (batch, num_candidates)."""
        item_np = items.numpy()
        scores = self.item_scores[item_np]
        return torch.from_numpy(scores).float()
