"""
ItemKNN baseline using scipy sparse cosine similarity.

Standard item-based k-nearest-neighbor collaborative filtering.
Computes item-item cosine similarity from the sparse user-item matrix, then predicts scores via weighted similarity to a user's past items.

Reference: Sarwar et al. (2001), "Item-Based Collaborative Filtering Recommendation Algorithms", WWW 2001.
"""

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:
    """Item-based KNN using sparse cosine similarity."""

    def __init__(self, k: int = 20, n_users: int = 0, n_items: int = 0):
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.sim = None          # sparse item-item similarity
        self.user_item = None    # sparse user-item matrix

    def fit(self, train_df: pd.DataFrame, n_users: int = 0, n_items: int = 0):
        """Fit ItemKNN: build sparse interaction matrix and item similarity."""
        users = train_df["user_id"].values
        items = train_df["item_id"].values

        # Use explicit ratings if available, else binary
        if "rating" in train_df.columns:
            data = train_df["rating"].values.astype(np.float32)
        else:
            data = np.ones(len(users), dtype=np.float32)

        if n_users > 0:
            self.n_users = n_users
        elif self.n_users == 0:
            self.n_users = int(users.max()) + 1

        if n_items > 0:
            self.n_items = n_items
        elif self.n_items == 0:
            self.n_items = int(items.max()) + 1

        self.user_item = sparse.csr_matrix(
            (data, (users, items)),
            shape=(self.n_users, self.n_items),
        )
        print(f"  ItemKNN: built {self.n_users}x{self.n_items} matrix, "
              f"nnz={self.user_item.nnz:,}")

        # Item-item cosine similarity
        item_feat = self.user_item.T.tocsr()
        print(f"  Computing item-item cosine similarity...")
        full_sim = cosine_similarity(item_feat, dense_output=False)

        # Keep only top-k neighbors per item
        print(f"  Sparsifying to top-{self.k} neighbors per item...")
        rows, cols, vals = [], [], []
        full_sim_csr = full_sim.tocsr()
        for i in range(self.n_items):
            row = full_sim_csr.getrow(i).toarray().ravel()
            row[i] = 0  # exclude self-similarity
            if self.k < len(row):
                top_k_idx = np.argpartition(row, -self.k)[-self.k:]
                for j in top_k_idx:
                    if row[j] > 0:
                        rows.append(i)
                        cols.append(j)
                        vals.append(row[j])
            else:
                nz = np.nonzero(row)[0]
                for j in nz:
                    rows.append(i)
                    cols.append(j)
                    vals.append(row[j])

        self.sim = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_items, self.n_items),
        )
        print(f"  ItemKNN: fit done, sim nnz={self.sim.nnz:,}")
        return self

    def predict_batch(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for candidate items (used by evaluation framework).
        users: (batch_size,)
        items: (batch_size, num_candidates)
        Returns: (batch_size, num_candidates) scores
        """
        batch_size, num_candidates = items.shape
        scores = np.zeros((batch_size, num_candidates), dtype=np.float32)

        for b in range(batch_size):
            u = users[b].item()
            user_vec = self.user_item.getrow(u)
            if user_vec.nnz == 0:
                continue
            candidate_ids = items[b].numpy()
            sim_rows = self.sim[candidate_ids]
            candidate_scores = sim_rows.dot(user_vec.T).toarray().ravel()
            scores[b] = candidate_scores

        return torch.from_numpy(scores).float()

    def predict(self, user_ids, item_ids):
        """Score specific (user, item) pairs. Accepts scalars or arrays."""
        scalar = np.isscalar(user_ids)
        if scalar:
            user_ids = [user_ids]
            item_ids = [item_ids]

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        scores = np.zeros(len(user_ids), dtype=np.float32)

        for idx in range(len(user_ids)):
            u, i = int(user_ids[idx]), int(item_ids[idx])
            if u >= self.n_users or i >= self.n_items:
                continue
            user_vec = self.user_item.getrow(u)
            sim_to_history = self.sim[i, :].toarray().flatten()
            history_items = user_vec.indices
            scores[idx] = sim_to_history[history_items].sum()

        if scalar:
            return float(scores[0])
        return scores

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        """Return top-k recommended item IDs for a user."""
        if user_id >= self.n_users:
            return []

        user_vec = self.user_item[user_id]
        scores = self.sim.T.dot(user_vec.T).toarray().flatten()

        if exclude_seen:
            seen = user_vec.indices
            scores[seen] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return top_k.tolist()
