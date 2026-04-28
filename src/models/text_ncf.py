"""
Review-text-enhanced Neural Collaborative Filtering (TextNCF).

Two-branch hybrid: GMF (collaborative) + text (sentence embeddings) fused through a small MLP. Text embeddings are pre-computed offline
with all-MiniLM-L6-v2 and loaded as registered buffers so the model is compatible with the shared evaluate_ranking(model, loader) interface.

Architecture:
    GMF branch:  user_emb ⊙ item_emb  →  (embed_dim,)
    Text branch: proj(user_text) ⊙ proj(item_text)  →  (text_proj_dim,)
    Fusion:      concat → MLP → sigmoid → score

"""

import torch
import torch.nn as nn

import numpy as np


class TextNCF(nn.Module):

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

        # fusion MLP
        fusion_dim = 0
        if use_gmf:
            fusion_dim += embed_dim
        if use_text:
            fusion_dim += text_proj_dim

        layers = []
        prev = fusion_dim
        for h in mlp_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        # text embedding buffers - set via load_text_embeddings()
        # registered as buffers so they move to GPU with .to(device)
        self.register_buffer("_user_text_emb", torch.zeros(1, text_dim))
        self.register_buffer("_item_text_emb", torch.zeros(1, text_dim))
        self._text_loaded = False

    def load_text_embeddings(self, user_emb_path: str, item_emb_path: str):
        """Load pre-computed text embeddings from .npy files."""
        u = np.load(user_emb_path)
        i = np.load(item_emb_path)
        self._user_text_emb = torch.from_numpy(u)
        self._item_text_emb = torch.from_numpy(i)
        self._text_loaded = True

    def set_text_embeddings(self, user_emb: torch.Tensor, item_emb: torch.Tensor):
        """Set text embeddings directly from tensors."""
        self._user_text_emb = user_emb
        self._item_text_emb = item_emb
        self._text_loaded = True

    def forward(self, user_ids, item_ids):
        """Score (user, item) pairs.

        Compatible with shared evaluate_ranking: just pass user_ids and item_ids.
        Text embeddings are looked up from the registered buffers.
        """
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
        return self.mlp(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
