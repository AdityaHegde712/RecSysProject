"""
Sub-Rating Decomposition TextNCF.

HotelRec has 6 aspect-level sub-ratings:
  Service, Cleanliness, Location, Value, Rooms, Sleep Quality

Instead of predicting a single score, this model predicts each sub-rating separately and combines them with learned per-user attention weights.
The intuition: different travelers care about different aspects (business travelers care about WiFi/location, families care about rooms/cleanliness).

Architecture:
    Same two-branch fusion as TextNCF for the shared representation.
    Then 6 parallel sub-rating heads (128 → 32 → 1 each).
    A user attention network learns weights over the 6 aspects.
    Final score = weighted sum of sub-rating predictions.

Training:
    MSE on each sub-rating + BPR on the combined score.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


NUM_ASPECTS = 6


class SubratingHead(nn.Module):
    """Single aspect prediction head: hidden → 32 → 1."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TextNCFSubrating(nn.Module):

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
        num_aspects: int = NUM_ASPECTS,
        aspect_hidden: int = 32,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_gmf = use_gmf
        self.use_text = use_text
        self.num_aspects = num_aspects

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

        # shared fusion MLP
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
        self.hidden_dim = prev  # output dim of shared MLP

        # 6 sub-rating prediction heads
        self.aspect_heads = nn.ModuleList([
            SubratingHead(prev, aspect_hidden)
            for _ in range(num_aspects)
        ])

        # user attention network: learns which aspects matter per user
        # input: user collaborative embedding (or a separate small embedding)
        self.user_attn_emb = nn.Embedding(num_users, 32)
        nn.init.normal_(self.user_attn_emb.weight, std=0.01)
        self.attn_net = nn.Sequential(
            nn.Linear(32, num_aspects),
            # softmax applied in forward
        )

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
        """Shared encoding through fusion MLP."""
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

    def predict_aspects(self, user_ids, item_ids):
        """Predict all 6 sub-ratings. Returns (batch, 6) tensor."""
        h = self._encode(user_ids, item_ids)
        preds = torch.stack([head(h) for head in self.aspect_heads], dim=1)
        return preds

    def get_attention_weights(self, user_ids):
        """Get per-user attention weights over aspects. Returns (batch, 6)."""
        u_attn = self.user_attn_emb(user_ids)
        logits = self.attn_net(u_attn)
        return F.softmax(logits, dim=-1)

    def forward(self, user_ids, item_ids):
        """Score (user, item) pairs.

        Combined score = attention-weighted sum of sub-rating predictions.
        Compatible with shared evaluate_ranking interface.
        """
        aspect_preds = self.predict_aspects(user_ids, item_ids)  # (B, 6)
        attn_weights = self.get_attention_weights(user_ids)       # (B, 6)

        # weighted sum across aspects
        score = (aspect_preds * attn_weights).sum(dim=1)
        return score

    def forward_detailed(self, user_ids, item_ids):
        """Return combined score, aspect predictions, and attention weights.

        Used during training to compute both BPR and MSE losses.
        """
        aspect_preds = self.predict_aspects(user_ids, item_ids)
        attn_weights = self.get_attention_weights(user_ids)
        score = (aspect_preds * attn_weights).sum(dim=1)
        return score, aspect_preds, attn_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
