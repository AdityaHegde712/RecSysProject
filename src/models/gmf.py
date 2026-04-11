import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    Generalized Matrix Factorization (He et al., 2017).
    Element-wise product of user/item embeddings -> linear -> sigmoid.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = u * i  # element-wise product
        return self.sigmoid(self.out(x)).squeeze(-1)
