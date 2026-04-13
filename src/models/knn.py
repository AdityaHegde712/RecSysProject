"""
Item-based K-Nearest Neighbors collaborative filtering (Sarwar et al., 2001).

Builds a sparse user-item interaction matrix, computes item-item cosine
similarity, and recommends items similar to those the user has already
interacted with. No neural network — just linear algebra.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:
    """
    Item-based collaborative filtering with cosine similarity.

    For each item, we keep only the top-k most similar neighbors to avoid
    blowing up memory on large catalogs. Recommendations are scored by
    summing similarities to items in the user's history.
    """

    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self.sim_matrix = None          # (num_items, num_items) sparse
        self.interaction_matrix = None  # (num_users, num_items) sparse
        self.num_users = 0
        self.num_items = 0

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int):
        """
        Build the item-item similarity matrix from training interactions.

        Args:
            train_df: DataFrame with columns 'user_id' and 'item_id'
            num_users: total number of users (for matrix dimensions)
            num_items: total number of items (for matrix dimensions)
        """
        self.num_users = num_users
        self.num_items = num_items

        # Build sparse binary interaction matrix (users x items)
        users = train_df["user_id"].values
        items = train_df["item_id"].values
        data = np.ones(len(users), dtype=np.float32)
        self.interaction_matrix = sparse.csr_matrix(
            (data, (users, items)), shape=(num_users, num_items)
        )

        # Compute item-item cosine similarity in batches to avoid OOM.
        # cosine_similarity on sparse input is efficient, but the output
        # is dense (num_items x num_items) which can be huge. We process
        # in chunks and keep only top-k per row.
        batch_size = 1000
        sim_rows = []
        sim_cols = []
        sim_vals = []

        item_matrix = self.interaction_matrix.T.tocsr()  # (items x users)

        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            # similarity of items[start:end] against all items
            chunk_sim = cosine_similarity(
                item_matrix[start:end], item_matrix
            )  # (chunk_size, num_items)

            for local_idx in range(chunk_sim.shape[0]):
                global_idx = start + local_idx
                row = chunk_sim[local_idx]
                # zero out self-similarity
                row[global_idx] = 0.0

                # keep only top-k neighbors
                if self.k < num_items:
                    top_k_idx = np.argpartition(row, -self.k)[-self.k:]
                    top_k_vals = row[top_k_idx]
                    # filter out zeros
                    mask = top_k_vals > 0
                    top_k_idx = top_k_idx[mask]
                    top_k_vals = top_k_vals[mask]
                else:
                    nonzero = np.nonzero(row)[0]
                    top_k_idx = nonzero
                    top_k_vals = row[nonzero]

                sim_rows.extend([global_idx] * len(top_k_idx))
                sim_cols.extend(top_k_idx.tolist())
                sim_vals.extend(top_k_vals.tolist())

        self.sim_matrix = sparse.csr_matrix(
            (sim_vals, (sim_rows, sim_cols)),
            shape=(num_items, num_items),
        )

        n_nonzero = self.sim_matrix.nnz
        print(f"  ItemKNN fitted: {num_users:,} users, {num_items:,} items, "
              f"{n_nonzero:,} similarity entries (k={self.k})")

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        """
        Return top-k recommended item IDs for a user.

        Scores each item by summing its similarity to items the user
        has already interacted with.
        """
        if user_id >= self.num_users:
            return []

        # user's interaction vector: (1, num_items) sparse
        user_vec = self.interaction_matrix[user_id]

        # score all items: sum of similarities to user's history
        # scores = sim_matrix.T @ user_vec.T → (num_items, 1)
        scores = self.sim_matrix.T.dot(user_vec.T).toarray().flatten()

        if exclude_seen:
            seen = user_vec.indices
            scores[seen] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return top_k.tolist()

    def predict(self, user_ids, item_ids):
        """
        Score specific (user, item) pairs.

        Can accept:
        - Two scalars: predict(u, i) → float
        - Two arrays: predict([u1,u2], [i1,i2]) → np.ndarray
        """
        scalar = np.isscalar(user_ids)
        if scalar:
            user_ids = [user_ids]
            item_ids = [item_ids]

        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        scores = np.zeros(len(user_ids), dtype=np.float32)

        for idx in range(len(user_ids)):
            u, i = int(user_ids[idx]), int(item_ids[idx])
            if u >= self.num_users or i >= self.num_items:
                continue
            user_vec = self.interaction_matrix[u]
            # similarity of item i to all items the user interacted with
            sim_to_history = self.sim_matrix[i, :].toarray().flatten()
            history_items = user_vec.indices
            scores[idx] = sim_to_history[history_items].sum()

        if scalar:
            return float(scores[0])
        return scores
