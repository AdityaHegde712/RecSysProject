"""
LightGCN-HG: LightGCN on a hotel-metadata heterogeneous graph.

Extends vanilla LightGCN's (user, item) bipartite graph with up to three extra node tiers drawn from hotel_url metadata:

    - g_id     : TripAdvisor leaf-location id (one city / neighbourhood)
    - region   : last two underscore tokens of the location slug
    - country  : last one underscore token of the location slug

Each hotel connects to its tier nodes with an unweighted edge. Tier nodes connect nowhere else, they act as pivot hubs that let hotels in the same
area exchange signal during propagation even when they have no direct user overlap. Symmetric normalization handles the resulting degree imbalance.

The scoring API is identical to vanilla LightGCN (user-item dot product). Tier embeddings exist only to influence user/item embeddings through
propagation. Same BPR training loop as LightGCN, same eval protocol (HR@k, NDCG@k under 1-vs-99). 
The only model-level change is the bigger adjacency and embedding table.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.graph.hetero_adj import build_hg_norm_adj_scipy


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_hg_norm_adj(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    hotel_meta: pd.DataFrame,
    n_users: int,
    n_items: int,
    tiers: Iterable[str] = ("g_id", "region_slug", "country_slug"),
) -> tuple[torch.sparse.Tensor, dict]:
    """Torch-sparse wrapper around ``build_hg_norm_adj_scipy``.

    Nodes are laid out as a single contiguous block, concatenated in this
    order: [users | items | tier_1 | tier_2 | ...]. Each tier column in
    ``hotel_meta`` contributes its own contiguous range.

    Edges included:
        * user <-> item (from interactions; both directions for symmetry)
        * item <-> tier node (one edge per hotel per selected tier)

    Returns:
        A_hat : (N, N) torch sparse COO tensor (coalesced).
        meta  : dict with node counts, offsets, and per-tier id maps.
    """
    A_hat, meta = build_hg_norm_adj_scipy(
        user_ids, item_ids, hotel_meta, n_users, n_items, tiers
    )
    indices = torch.from_numpy(np.vstack([A_hat.row, A_hat.col])).long()
    values = torch.from_numpy(A_hat.data).float()
    A_hat_t = torch.sparse_coo_tensor(
        indices, values, (meta["n_total"], meta["n_total"])
    ).coalesce()
    return A_hat_t, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LightGCNHG(nn.Module):
    """LightGCN over the metadata-augmented heterogeneous graph.

    The only structural difference from vanilla LightGCN is a single
    embedding table covering all node types, and propagation over the
    full (users + items + tier nodes) adjacency. Scoring uses only user
    and item slices of the propagated embeddings.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        graph_meta: dict,
        embed_dim: int = 256,
        num_layers: int = 1,
        adj_hat: torch.sparse.Tensor | None = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_total = int(graph_meta["n_total"])
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.graph_meta = graph_meta

        # One embedding table for every node type. Slicing is done with
        # offsets from graph_meta at propagation / scoring time.
        self.node_emb = nn.Embedding(self.num_total, embed_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.register_buffer("adj_hat", adj_hat)
        self._cached_u: torch.Tensor | None = None
        self._cached_i: torch.Tensor | None = None

    # --- propagation -------------------------------------------------------

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K layers of graph convolution over all nodes.

        Returns only the user and item embedding slices since scoring
        never uses the tier nodes directly.
        """
        all_emb = self.node_emb.weight
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj_hat, all_emb)
            embs.append(all_emb)
        final = torch.stack(embs, dim=1).mean(dim=1)
        users = final[: self.num_users]
        items = final[self.num_users : self.num_users + self.num_items]
        return users, items

    # --- eval-time cache management ---------------------------------------

    def cache_embeddings(self) -> None:
        with torch.no_grad():
            self._cached_u, self._cached_i = self.propagate()

    def invalidate_cache(self) -> None:
        self._cached_u = None
        self._cached_i = None

    def train(self, mode: bool = True):
        if mode:
            self.invalidate_cache()
        return super().train(mode)

    # --- scoring -----------------------------------------------------------

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
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
        """BPR scoring -- one propagation pass, returns layer-0 embeddings
        for L2 regularization (matching LightGCN paper convention).

        Only the layer-0 user and item embeddings are penalized; tier-node
        embeddings are left free to absorb the geographic signal without
        shrinkage.
        """
        users_emb, items_emb = self.propagate()
        u = users_emb[user_ids]
        pi = items_emb[pos_items]
        ni = items_emb[neg_items]
        pos_scores = (u * pi).sum(dim=-1)
        neg_scores = (u * ni).sum(dim=-1)

        u_offset = self.graph_meta["offsets"]["user"]
        i_offset = self.graph_meta["offsets"]["item"]
        u0 = self.node_emb(user_ids + u_offset)
        pi0 = self.node_emb(pos_items + i_offset)
        ni0 = self.node_emb(neg_items + i_offset)
        return pos_scores, neg_scores, (u0, pi0, ni0)
