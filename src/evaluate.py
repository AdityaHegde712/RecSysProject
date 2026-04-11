"""
Evaluation entry point for HotelRec — GMF baseline.

Loads a trained GMF checkpoint and evaluates it on the test split using
the leave-one-out protocol (1 positive + 99 negatives per user).
Reports HR@k and NDCG@k for k=5,10,20.

Usage:
    python -m src.evaluate --config configs/gmf.yaml --kcore 20
"""

import argparse
import gc
import os
from pathlib import Path

import torch

from src.data.dataset import get_dataloaders, get_n_users_items
from src.metrics.ranking import evaluate_ranking
from src.models.common import build_model
from src.utils.io import load_config, load_checkpoint
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, kcore_dir, num_users, num_items, device):
    """Load checkpoint and evaluate GMF on the test set."""
    paths = config.get("paths", {})
    ckpt_dir = paths.get("checkpoint_dir", "results/gmf")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        print("Train the model first: python -m src.train --config configs/gmf.yaml")
        return None

    # load model
    model = build_model(config, num_users, num_items)
    model, epoch = load_checkpoint(ckpt_path, model)
    model = model.to(device)
    print(f"Loaded GMF (epoch {epoch}) from {ckpt_path}")

    # build eval dataloader
    eval_cfg = config.get("evaluation", {})
    neg_cfg = config.get("negative_sampling", {})
    k_values = eval_cfg.get("top_k", [5, 10, 20])

    loaders = get_dataloaders(
        kcore_dir,
        batch_size=eval_cfg.get("batch_size", 256),
        eval_negatives=eval_cfg.get("num_negatives",
                                    neg_cfg.get("num_negatives", 99)),
        seed=config.get("split", {}).get("seed", 42),
    )

    results = evaluate_ranking(model, loaders["test"],
                               k_values=k_values, device=str(device))

    # print results
    print(f"\n{'=' * 55}")
    print(f"  GMF — Top-K Recommendation")
    print(f"{'=' * 55}")
    for metric, val in sorted(results.items()):
        print(f"    {metric}: {val:.4f}")
    print(f"{'=' * 55}")

    # cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GMF on HotelRec")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Data config YAML")
    parser.add_argument("--kcore", type=int, default=20,
                        help="K-core value (5 or 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    data_cfg = (load_config(args.data_config)
                if Path(args.data_config).exists() else {})

    kcore_dir = os.path.join(
        data_cfg.get("dataset", {}).get("processed_dir", "data/processed"),
        f"kcore_{args.kcore}",
    )
    if not os.path.isdir(kcore_dir):
        raise FileNotFoundError(
            f"Processed data not found: {kcore_dir}\n"
            f"Run: python -m src.data.preprocess --kcore {args.kcore}")

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: kcore={args.kcore}, {num_users:,} users, {num_items:,} items")

    evaluate(config, kcore_dir, num_users, num_items, device)


if __name__ == "__main__":
    main()
