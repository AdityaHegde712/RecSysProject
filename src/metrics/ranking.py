import math

import numpy as np
from tqdm import tqdm

from src.data.dataset import load_split, get_user_positive_items


def hit_ratio(ranked_list, ground_truth, k: int) -> float:
    """1 if the ground-truth item appears in the top-k ranked list, else 0."""
    return float(ground_truth in ranked_list[:k])


def ndcg(ranked_list, ground_truth, k: int) -> float:
    """NDCG@k for a single user with one relevant item."""
    top_k = ranked_list[:k]
    for i, item in enumerate(top_k):
        if item == ground_truth:
            return 1.0 / math.log2(i + 2)  # i is 0-indexed, DCG uses log2(rank+1)
    return 0.0


def evaluate_ranking(
    model,
    kcore_dir: str,
    split: str = "test",
    k_values: list[int] = None,
    num_negatives: int = 99,
    seed: int = 42,
) -> dict[str, float]:
    """
    Leave-one-out evaluation for ItemKNN.

    For each user in the split, takes their positive item and samples
    num_negatives random items the user hasn't interacted with. Scores
    all candidates and computes HR@k and NDCG@k.

    Args:
        model: fitted ItemKNN model
        kcore_dir: path to the k-core processed directory
        split: which split to evaluate ('val' or 'test')
        k_values: list of k values for HR@k and NDCG@k
        num_negatives: number of negative samples per positive
        seed: random seed for negative sampling
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # load the split and user positive items
    eval_df = load_split(kcore_dir, split)
    user_pos_all = get_user_positive_items(kcore_dir)

    rng = np.random.RandomState(seed)
    n_items = model.num_items

    metrics = {f"{m}@{k}": [] for k in k_values for m in ("HR", "NDCG")}

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df),
                       desc=f"eval ({split})", leave=False):
        u = int(row["user_id"])
        pos_item = int(row["item_id"])
        pos_set = user_pos_all.get(u, set())

        # sample negatives
        negs = []
        while len(negs) < num_negatives:
            j = rng.randint(0, n_items)
            if j not in pos_set and j != pos_item:
                negs.append(j)

        # score all candidates: positive first, then negatives
        candidates = [pos_item] + negs
        scores = model.predict(
            [u] * len(candidates),
            candidates,
        )

        # rank by descending score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_items = [candidates[i] for i in ranked_indices]

        for k in k_values:
            metrics[f"HR@{k}"].append(hit_ratio(ranked_items, pos_item, k))
            metrics[f"NDCG@{k}"].append(ndcg(ranked_items, pos_item, k))

    return {key: float(np.mean(vals)) for key, vals in metrics.items()}
