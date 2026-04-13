"""
Ranking evaluation metrics: HR@k and NDCG@k.

Uses leave-one-out protocol: 1 positive + 99 negatives per test user.
Supports both neural models (via forward()) and non-neural (via predict_batch()).
"""

import math

import numpy as np
import torch
from tqdm import tqdm


def hit_ratio(ranked_list, ground_truth, k: int) -> float:
    """1 if ground_truth appears in the top-k ranked list, else 0."""
    return float(ground_truth in ranked_list[:k])


def ndcg(ranked_list, ground_truth, k: int) -> float:
    """NDCG@k for a single user with one relevant item."""
    top_k = ranked_list[:k]
    for i, item in enumerate(top_k):
        if item == ground_truth:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate_ranking(
    model,
    test_loader,
    k_values: list[int] = None,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Leave-one-out evaluation: 1 positive + 99 negatives per test user.
    Returns dict like {'HR@5': 0.32, 'NDCG@5': 0.18, ...}.

    Works with:
    - Neural models (nn.Module with forward(users, items))
    - Non-neural models (with predict_batch(users, items))
    """
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {f"{m}@{k}": [] for k in k_values for m in ("HR", "NDCG")}

    model_is_neural = hasattr(model, "parameters")
    if model_is_neural:
        model.eval()

    with torch.no_grad():
        for users, items, labels in tqdm(test_loader, desc="eval", leave=False):
            batch_size = users.size(0)
            num_candidates = items.size(1)

            if model_is_neural:
                u_flat = users.unsqueeze(1).expand(-1, num_candidates).reshape(-1).to(device)
                i_flat = items.reshape(-1).to(device)
                scores = model(u_flat, i_flat).reshape(batch_size, num_candidates)
            else:
                scores = model.predict_batch(users, items)

            _, indices = torch.sort(scores, dim=1, descending=True)
            indices = indices.cpu()  # ensure CPU for indexing

            for b in range(batch_size):
                ranked = items[b][indices[b]].tolist()
                gt_item = items[b, 0].item()

                for k in k_values:
                    metrics[f"HR@{k}"].append(hit_ratio(ranked, gt_item, k))
                    metrics[f"NDCG@{k}"].append(ndcg(ranked, gt_item, k))

    return {key: float(np.mean(vals)) for key, vals in metrics.items()}
