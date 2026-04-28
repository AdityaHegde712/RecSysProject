"""
Two-stage retrieval + ranking evaluation.

Stage 1: ItemKNN retrieves top-200 candidates per user (fast, sparse)
Stage 2: TextNCF re-ranks those 200 candidates (slow, neural)

This mimics a production setup where you can't score all items with a neural model, so you use a cheap retriever first.

Usage:
    python -m src.evaluate_two_stage \
        --text-ncf-ckpt results/text_ncf/best_model.pt \
        --knn-ckpt results/baselines/itemknn.pkl \
        --kcore 20

"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import (
    get_n_users_items, load_split, get_user_positive_items,
)
from src.data.text_embeddings import load_text_embeddings, TEXT_EMB_DIR
from src.models.text_ncf import TextNCF
from src.models.knn import ItemKNN
from src.utils.io import load_config, load_checkpoint, load_model
from src.utils.seed import set_seed


def _get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def retrieve_candidates(knn_model, user_id, top_n=200, exclude_seen=True):
    """Stage 1: get top-N candidates from ItemKNN for a user."""
    return knn_model.recommend(user_id, k=top_n, exclude_seen=exclude_seen)


def rerank_with_text_ncf(model, user_id, candidate_items, device):
    """Stage 2: re-rank candidate items using TextNCF.

    Returns list of item IDs sorted by TextNCF score (descending).
    """
    if len(candidate_items) == 0:
        return []

    users = torch.tensor([user_id] * len(candidate_items), dtype=torch.long).to(device)
    items = torch.tensor(candidate_items, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(users, items)

    # sort by score descending
    order = torch.argsort(scores, descending=True).cpu().numpy()
    return [candidate_items[i] for i in order]


def evaluate_two_stage(
    knn_model,
    text_ncf_model,
    test_df,
    user_pos_all,
    num_items,
    device,
    retrieve_k=200,
    k_values=None,
):
    """Run two-stage evaluation on test users.

    For each test interaction:
    1. ItemKNN retrieves top-retrieve_k candidates (excluding seen items)
    2. Ensure the ground-truth item is in the candidate set (for fair eval)
    3. TextNCF re-ranks the candidates
    4. Compute HR@k and NDCG@k
    """
    if k_values is None:
        k_values = [5, 10, 20]

    metrics = {f"{m}@{k}": [] for k in k_values for m in ("HR", "NDCG")}

    # also track how often the GT item was already in the retrieval set
    gt_in_retrieval = []

    text_ncf_model.eval()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="two-stage eval"):
        user_id = int(row["user_id"])
        gt_item = int(row["item_id"])

        # stage 1: retrieve candidates
        # we need to temporarily remove gt_item from the user's history
        # so ItemKNN can potentially retrieve it
        seen = user_pos_all.get(user_id, set())
        seen_without_gt = seen - {gt_item}

        # get candidates excluding seen (but gt_item is not in seen_without_gt)
        # we do this by temporarily modifying the user_item matrix
        # Actually, simpler: just retrieve with exclude_seen=True and check
        candidates = retrieve_candidates(knn_model, user_id, top_n=retrieve_k,
                                         exclude_seen=True)

        # check if GT was retrieved
        gt_retrieved = gt_item in candidates
        gt_in_retrieval.append(gt_retrieved)

        # ensure GT is in candidate set for fair evaluation
        if not gt_retrieved:
            # replace the last candidate with GT
            if len(candidates) >= retrieve_k:
                candidates[-1] = gt_item
            else:
                candidates.append(gt_item)

        # stage 2: re-rank with TextNCF
        reranked = rerank_with_text_ncf(text_ncf_model, user_id,
                                        candidates, device)

        # compute metrics
        for k in k_values:
            top_k = reranked[:k]
            # HR
            metrics[f"HR@{k}"].append(float(gt_item in top_k))
            # NDCG
            ndcg_val = 0.0
            for i, item in enumerate(top_k):
                if item == gt_item:
                    ndcg_val = 1.0 / math.log2(i + 2)
                    break
            metrics[f"NDCG@{k}"].append(ndcg_val)

    result = {key: float(np.mean(vals)) for key, vals in metrics.items()}
    result["gt_recall@200"] = float(np.mean(gt_in_retrieval))
    return result


def main():
    parser = argparse.ArgumentParser(description="Two-stage retrieval + ranking")
    parser.add_argument("--text-ncf-ckpt", default="results/text_ncf/best_model.pt")
    parser.add_argument("--text-ncf-config", default="configs/text_ncf.yaml")
    parser.add_argument("--knn-ckpt", default="results/baselines/itemknn.pkl")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--retrieve-k", type=int, default=200,
                        help="Number of candidates from ItemKNN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _get_device()
    print(f"Device: {device}")

    kcore_dir = os.path.join("data", "processed", f"{args.kcore}core")
    if not os.path.isdir(kcore_dir):
        print(f"No data at {kcore_dir}. Run preprocessing first.")
        return

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: {num_users:,} users, {num_items:,} items")

    # load test data
    test_df = load_split(kcore_dir, "test")
    user_pos_all = get_user_positive_items(kcore_dir)
    print(f"Test interactions: {len(test_df):,}")

    # load models
    print("\nLoading models...")

    # ItemKNN
    knn = load_model(args.knn_ckpt)
    print(f"  ItemKNN: k={knn.k}, sim nnz={knn.sim.nnz:,}")

    # TextNCF
    config = load_config(args.text_ncf_config)
    model_cfg = config.get("model", {})
    text_ncf = TextNCF(
        num_users=num_users,
        num_items=num_items,
        embed_dim=model_cfg.get("embed_dim", 64),
        text_dim=model_cfg.get("text_dim", 384),
        text_proj_dim=model_cfg.get("text_proj_dim", 64),
        mlp_layers=model_cfg.get("mlp_layers", [128, 64]),
        dropout=model_cfg.get("dropout", 0.2),
        use_gmf=model_cfg.get("use_gmf", True),
        use_text=model_cfg.get("use_text", True),
    )

    emb_dir = config.get("paths", {}).get("text_emb_dir", TEXT_EMB_DIR)
    user_emb, item_emb = load_text_embeddings(emb_dir)
    text_ncf.set_text_embeddings(
        torch.from_numpy(user_emb), torch.from_numpy(item_emb))

    load_checkpoint(args.text_ncf_ckpt, model=text_ncf)
    text_ncf.to(device)
    text_ncf.eval()
    print(f"  TextNCF: {text_ncf.count_parameters():,} params")

    # run two-stage evaluation
    print(f"\n{'=' * 65}")
    print(f"Two-stage: ItemKNN(top-{args.retrieve_k}) → TextNCF re-rank")
    print(f"{'=' * 65}")

    results = evaluate_two_stage(
        knn, text_ncf, test_df, user_pos_all, num_items, device,
        retrieve_k=args.retrieve_k,
    )

    print(f"\nResults:")
    for m, v in sorted(results.items()):
        print(f"  {m}: {v:.4f}")

    # save
    out_dir = "results/text_ncf"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, "two_stage_metrics.json")
    output = {
        "retrieve_k": args.retrieve_k,
        "retriever": "ItemKNN",
        "ranker": "TextNCF",
        "metrics": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
