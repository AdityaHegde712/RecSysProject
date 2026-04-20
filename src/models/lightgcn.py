"""
LightGCN (He et al., SIGIR 2020) for implicit-feedback recommendation.

https://arxiv.org/abs/2002.02126

Simplified graph convolution over the user-item bipartite graph: no feature
transformation, no non-linearity, just weighted neighborhood aggregation.
Final embeddings are the mean across layers. Trained with BPR loss.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def build_norm_adj(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
) -> torch.sparse.Tensor:
    """Build the symmetrically-normalized adjacency matrix A_hat as torch sparse.

    A is the (N+M) x (N+M) bipartite adjacency with R in the upper-right block
    and R^T in the lower-left block. A_hat = D^{-1/2} A D^{-1/2}.
    """
    n = n_users + n_items
    # Build COO of A: user -> item (+ n_users offset)
    rows = np.concatenate([user_ids, item_ids + n_users])
    cols = np.concatenate([item_ids + n_users, user_ids])
    data = np.ones(len(rows), dtype=np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    # Degree
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    A_hat = A_hat.tocoo()

    indices = torch.from_numpy(np.vstack([A_hat.row, A_hat.col])).long()
    values = torch.from_numpy(A_hat.data).float()
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class LightGCN(nn.Module):
    """LightGCN bipartite graph convolution.

    Args:
        num_users: number of users.
        num_items: number of items.
        embed_dim: embedding dimension.
        num_layers: number of propagation layers (K).
        adj_hat: precomputed sparse normalized adjacency matrix (N+M, N+M).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        num_layers: int = 3,
        adj_hat: torch.sparse.Tensor = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.register_buffer("adj_hat", adj_hat)
        self._cached_u = None
        self._cached_i = None

    # --- propagation ---

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K layers of graph convolution. Returns (users, items) embeddings."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj_hat, all_emb)
            embs.append(all_emb)
        final = torch.stack(embs, dim=1).mean(dim=1)
        return final[: self.num_users], final[self.num_users :]

    # --- eval-time cache management ---

    def cache_embeddings(self) -> None:
        """Precompute propagated embeddings for fast batched eval."""
        with torch.no_grad():
            self._cached_u, self._cached_i = self.propagate()

    def invalidate_cache(self) -> None:
        self._cached_u = None
        self._cached_i = None

    def train(self, mode: bool = True):
        # Invalidate cache when switching to training mode.
        if mode:
            self.invalidate_cache()
        return super().train(mode)

    # --- scoring API ---

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score (user, item) pairs. Uses cache if in eval mode and primed."""
        if not self.training and self._cached_u is not None:
            u = self._cached_u[user_ids]
            i = self._cached_i[item_ids]
        else:
            users_emb, items_emb = self.propagate()
            u = users_emb[user_ids]
            i = items_emb[item_ids]
        return (u * i).sum(dim=-1)

    def score_triplet(
        self,
        user_ids: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Compute pos/neg scores in a single propagation pass (for BPR).

        Also returns the raw initial (layer-0) embeddings of the sampled users
        and items for L2 regularization as in the LightGCN paper.
        """
        users_emb, items_emb = self.propagate()
        u = users_emb[user_ids]
        pi = items_emb[pos_items]
        ni = items_emb[neg_items]
        pos_scores = (u * pi).sum(dim=-1)
        neg_scores = (u * ni).sum(dim=-1)

        u0 = self.user_emb(user_ids)
        pi0 = self.item_emb(pos_items)
        ni0 = self.item_emb(neg_items)
        return pos_scores, neg_scores, (u0, pi0, ni0)
