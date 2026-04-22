"""
SASRec (Kang & McAuley, ICDM 2018): Self-Attentive Sequential Recommendation.

https://arxiv.org/abs/1808.09781

Causal self-attention over a user's time-ordered item history. At each
position the model predicts the next item from the prefix up to that
point. Trained with a BPR-style per-position loss (one sampled negative).

Uses the same 1-vs-99 eval protocol as every other model in this repo:
score the positive + 99 negatives, rank by score, compute HR@k and
NDCG@k.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """Self-attentive sequential recommender.

    Args:
        n_items        : number of distinct items (vocab size excl. pad)
        embed_dim      : token + position embedding dimension
        max_seqlen     : maximum sequence length (sets position table size)
        num_heads      : attention heads
        num_layers     : transformer blocks
        dropout        : dropout applied to embeddings, attention, FFN
    """

    def __init__(
        self,
        n_items: int,
        embed_dim: int = 64,
        max_seqlen: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # +1 for pad id at index 0; item ids are +1-shifted upstream.
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seqlen, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seqlen = max_seqlen
        self.embed_dim = embed_dim

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # True where attention is NOT allowed. Upper triangular excluding diag.
        m = torch.ones(L, L, dtype=torch.bool, device=device).triu(diagonal=1)
        return m

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a (batch, seqlen) item-id sequence into per-position
        hidden states of shape (batch, seqlen, dim)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.embed_dropout(x)
        pad_mask = seq == 0  # (B, L)
        attn_mask = self._causal_mask(L, seq.device)
        # Zero out embeddings for pad tokens upfront (mask is redundant but
        # cheap).
        x = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=pad_mask,
        )
        return x

    def last_position(self, seq: torch.Tensor) -> torch.Tensor:
        """Return the hidden state of the LAST non-pad position per row.

        Since sequences are left-padded, the last non-pad is just the
        last position (index L-1) of the tensor. Shape: (B, dim).
        """
        x = self.encode(seq)
        return x[:, -1, :]

    def score_candidates(
        self, seq: torch.Tensor, cand_items: torch.Tensor
    ) -> torch.Tensor:
        """For each row, score ``cand_items`` against the last-position
        representation. cand_items: (B, C) with +1-shifted ids. Returns
        (B, C) scores.
        """
        h = self.last_position(seq)                 # (B, D)
        e = self.item_emb(cand_items)                # (B, C, D)
        return (h.unsqueeze(1) * e).sum(-1)          # (B, C)

    def forward(self, seq: torch.Tensor, pos_items: torch.Tensor,
                neg_items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """BPR training forward. pos_items / neg_items are (B,) or (B, K)."""
        h = self.last_position(seq)                  # (B, D)
        p = self.item_emb(pos_items)                 # (B, D) or (B, K, D)
        n = self.item_emb(neg_items)                 # (B, D) or (B, K, D)
        if p.dim() == 2:
            pos_score = (h * p).sum(-1)
            neg_score = (h * n).sum(-1) if n.dim() == 2 else (h.unsqueeze(1) * n).sum(-1).mean(-1)
        else:
            pos_score = (h.unsqueeze(1) * p).sum(-1).mean(-1)
            neg_score = (h.unsqueeze(1) * n).sum(-1).mean(-1)
        return pos_score, neg_score
