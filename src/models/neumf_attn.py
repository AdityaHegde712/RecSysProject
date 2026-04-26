"""
NeuMF with Attention-Weighted Sub-Ratings (Variant B).

Extends Neural Matrix Factorization (He et al., WWW 2017) with a lightweight
per-user attention mechanism over the six hotel sub-rating dimensions
(Service, Cleanliness, Location, Value, Rooms, Sleep Quality).

Architecture
------------
- **GMF branch**  : element-wise product of user/item GMF embeddings.
- **MLP branch**  : concatenation of user/item MLP embeddings, passed
  through a stack of linear → ReLU → Dropout layers.
- **Sub-rating attention** : learns a per-user attention vector (weights)
  over the N_ASPECTS = 6 sub-rating dimensions. The weighted average of
  a hotel's pre-computed mean sub-ratings is concatenated to the fusion
  layer as an extra "quality score."
- **Fusion**       : concat(GMF_out, MLP_out, quality_score) → linear → scalar.

Training uses BPR loss (one sampled negative per interaction).
Evaluation uses the shared 1-vs-99 ranking protocol (HR@k, NDCG@k).

Usage:
    from src.models.neumf_attn import NeuMF_Attn
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The six sub-rating dimensions used in this variant.
SUB_RATING_DIMS = ["service", "cleanliness", "location", "value", "rooms", "sleep_quality"]
N_ASPECTS = len(SUB_RATING_DIMS)


class NeuMF_Attn(nn.Module):
    """NeuMF with per-user attention over hotel sub-ratings.

    Args:
        n_users        : number of distinct users.
        n_items        : number of distinct items (hotels).
        gmf_dim        : embedding dimension for the GMF branch.
        mlp_dim        : embedding dimension *per side* for the MLP branch.
        mlp_layers     : hidden layer sizes for the MLP stack (e.g. [256, 128, 64]).
        dropout        : dropout probability applied after each MLP hidden layer.
        item_aspects   : float tensor of shape (n_items, N_ASPECTS) containing
                         pre-computed per-hotel mean sub-ratings (train-only).
                         NaNs should have been filled with global means before
                         passing here.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: list[int] = None,
        dropout: float = 0.2,
        item_aspects: torch.Tensor | None = None,
        use_attention: bool = True,
    ):
        super().__init__()

        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.use_attention = use_attention

        # ---- GMF branch embeddings ----------------------------------------
        self.gmf_user_emb = nn.Embedding(n_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_dim)

        # ---- MLP branch embeddings ----------------------------------------
        self.mlp_user_emb = nn.Embedding(n_users, mlp_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_dim)

        # ---- MLP stack ----------------------------------------------------
        mlp_input_dim = mlp_dim * 2  # concat(user, item)
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        mlp_output_dim = in_dim  # = mlp_layers[-1]

        # ---- Sub-rating attention -----------------------------------------
        # Per-user attention weights over N_ASPECTS sub-rating dimensions.
        # We project the user's GMF embedding to N_ASPECTS scores (before
        # softmax) so the attention is user-specific and learned end-to-end.
        # Only instantiated when use_attention=True so the vanilla ablation
        # drops the parameters entirely (cleaner than zeroing them at runtime).
        if use_attention:
            self.attn_proj = nn.Linear(gmf_dim, N_ASPECTS, bias=True)

            # Frozen (non-gradient) lookup table for item aspect vectors.
            # Registered as a buffer so it moves with .to(device) calls.
            if item_aspects is not None:
                assert item_aspects.shape == (n_items, N_ASPECTS), (
                    f"item_aspects shape mismatch: expected ({n_items}, {N_ASPECTS}), "
                    f"got {tuple(item_aspects.shape)}"
                )
                self.register_buffer("item_aspects", item_aspects.float())
            else:
                # Zero fallback if aspects not provided (graceful degradation).
                self.register_buffer(
                    "item_aspects", torch.zeros(n_items, N_ASPECTS, dtype=torch.float)
                )

        # ---- Fusion layer -------------------------------------------------
        # Concatenate GMF output (gmf_dim), MLP output (mlp_output_dim),
        # and - when the attention branch is on - the scalar quality score (1).
        fusion_in = gmf_dim + mlp_output_dim + (1 if use_attention else 0)
        self.fusion = nn.Linear(fusion_in, 1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (same convention as SASRec: N(0, 0.02))
    # ------------------------------------------------------------------
    def _init_weights(self):
        for emb in (self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb):
            nn.init.normal_(emb.weight, std=0.02)
        if self.use_attention:
            nn.init.xavier_uniform_(self.attn_proj.weight)
            nn.init.zeros_(self.attn_proj.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Core forward helpers
    # ------------------------------------------------------------------
    def _gmf_out(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Element-wise product of GMF branch embeddings. Shape: (B, gmf_dim)."""
        return self.gmf_user_emb(users) * self.gmf_item_emb(items)

    def _mlp_out(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """MLP branch output. Shape: (B, mlp_layers[-1])."""
        u = self.mlp_user_emb(users)
        i = self.mlp_item_emb(items)
        return self.mlp(torch.cat([u, i], dim=-1))

    def _quality_score(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Per-user attention-weighted hotel sub-rating quality score.

        1. Derive per-user attention weights from the GMF user embedding.
        2. Look up the item's pre-computed aspect vector.
        3. Return the dot product (scalar per row). Shape: (B, 1).
        """
        u_emb = self.gmf_user_emb(users)                    # (B, gmf_dim)
        attn_logits = self.attn_proj(u_emb)                 # (B, N_ASPECTS)
        attn_weights = F.softmax(attn_logits, dim=-1)       # (B, N_ASPECTS)

        aspects = self.item_aspects[items]                   # (B, N_ASPECTS)
        score = (attn_weights * aspects).sum(dim=-1, keepdim=True)  # (B, 1)
        return score

    def _score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Fused scalar relevance score for (user, item) pairs. Shape: (B,)."""
        gmf = self._gmf_out(users, items)          # (B, gmf_dim)
        mlp = self._mlp_out(users, items)          # (B, mlp_out)
        if self.use_attention:
            qs = self._quality_score(users, items)     # (B, 1)
            fused = torch.cat([gmf, mlp, qs], dim=-1)  # (B, gmf+mlp+1)
        else:
            fused = torch.cat([gmf, mlp], dim=-1)      # (B, gmf+mlp)
        return self.fusion(fused).squeeze(-1)      # (B,)

    # ------------------------------------------------------------------
    # Public API (mirrors sasrec.py conventions)
    # ------------------------------------------------------------------
    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BPR training forward.

        Args:
            users     : (B,) user ids.
            pos_items : (B,) positive item ids.
            neg_items : (B,) negative item ids.

        Returns:
            pos_scores, neg_scores - each of shape (B,).
        """
        pos_scores = self._score(users, pos_items)
        neg_scores = self._score(users, neg_items)
        return pos_scores, neg_scores

    def score_candidates(
        self, users: torch.Tensor, cand_items: torch.Tensor
    ) -> torch.Tensor:
        """Score a set of candidate items per user for the 1-vs-99 eval.

        Args:
            users      : (B,) user ids.
            cand_items : (B, C) candidate item ids (first column = positive).

        Returns:
            scores of shape (B, C).
        """
        B, C = cand_items.shape
        users_expanded = users.unsqueeze(1).expand(B, C).reshape(-1)  # (B*C,)
        items_flat = cand_items.reshape(-1)                            # (B*C,)
        scores_flat = self._score(users_expanded, items_flat)          # (B*C,)
        return scores_flat.view(B, C)
