"""
Multi-Task TextNCF: joint ranking + rating prediction.

Extends TextNCF with a rating prediction head so the model learns from
both implicit (BPR) and explicit (MSE on ratings) signals simultaneously.
The idea is that predicting ratings forces the model to learn finer-grained
preferences, which should help ranking too.

Architecture:
    Same two-branch fusion as TextNCF, but with two output heads:
    - Ranking head:  MLP → 1 (sigmoid, for BPR)
    - Rating head:   Linear(hidden, 1) (for MSE on 1-5 ratings)

Loss: alpha * BPR_loss + (1 - alpha) * MSE_loss

Pramod Yadav — CMPE 256, Spring 2026
"""

import torch
import torch.nn as nn

import numpy as np


class TextNCFMultiTask(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        text_dim: int = 384,
        text_proj_dim: int = 64,
        mlp_layers: list[int] = None,
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_text: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_gmf = use_gmf
        self.use_text = use_text

        if mlp_layers is None:
            mlp_layers = [128, 64]

        # GMF branch
        if use_gmf:
            self.user_emb = nn.Embedding(num_users, embed_dim)
            self.item_emb = nn.Embedding(num_items, embed_dim)
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)

        # text branch
        if use_text:
            self.user_text_proj = nn.Linear(text_dim, text_proj_dim)
            self.item_text_proj = nn.Linear(text_dim, text_proj_dim)

        # shared fusion MLP (everything except the final layer)
        fusion_dim = 0
        if use_gmf:
            fusion_dim += embed_dim
        if use_text:
            fusion_dim += text_proj_dim

        shared_layers = []
        prev = fusion_dim
        for h in mlp_layers:
            shared_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.shared_mlp = nn.Sequential(*shared_layers)

        # ranking head (for BPR)
        self.rank_head = nn.Linear(prev, 1)

        # rating prediction head (for MSE)
        self.rating_head = nn.Linear(prev, 1)

        # text embedding buffers
        self.register_buffer("_user_text_emb", torch.zeros(1, text_dim))
        self.register_buffer("_item_text_emb", torch.zeros(1, text_dim))
        self._text_loaded = False

    def load_text_embeddings(self, user_emb_path: str, item_emb_path: str):
        u = np.load(user_emb_path)
        i = np.load(item_emb_path)
        self._user_text_emb = torch.from_numpy(u)
        self._item_text_emb = torch.from_numpy(i)
        self._text_loaded = True

    def set_text_embeddings(self, user_emb: torch.Tensor, item_emb: torch.Tensor):
        self._user_text_emb = user_emb
        self._item_text_emb = item_emb
        self._text_loaded = True

    def _encode(self, user_ids, item_ids):
        """Shared encoding: returns the hidden representation before heads."""
        parts = []

        if self.use_gmf:
            u_emb = self.user_emb(user_ids)
            i_emb = self.item_emb(item_ids)
            parts.append(u_emb * i_emb)

        if self.use_text:
            u_text = self._user_text_emb[user_ids]
            i_text = self._item_text_emb[item_ids]
            u_proj = self.user_text_proj(u_text)
            i_proj = self.item_text_proj(i_text)
            parts.append(u_proj * i_proj)

        x = torch.cat(parts, dim=-1)
        return self.shared_mlp(x)

    def forward(self, user_ids, item_ids):
        """Score (user, item) pairs for ranking.

        Compatible with shared evaluate_ranking interface.
        Uses the ranking head output.
        """
        h = self._encode(user_ids, item_ids)
        return self.rank_head(h).squeeze(-1)

    def predict_rating(self, user_ids, item_ids):
        """Predict ratings for (user, item) pairs.

        Returns predicted ratings (not clamped — clamping done in loss).
        """
        h = self._encode(user_ids, item_ids)
        return self.rating_head(h).squeeze(-1)

    def forward_both(self, user_ids, item_ids):
        """Return both ranking scores and rating predictions.

        Used during training to avoid encoding twice.
        """
        h = self._encode(user_ids, item_ids)
        rank_score = self.rank_head(h).squeeze(-1)
        rating_pred = self.rating_head(h).squeeze(-1)
        return rank_score, rating_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
