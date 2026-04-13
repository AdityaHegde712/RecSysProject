"""
Evaluation entry point for HotelRec — ItemKNN baseline.

Loads a fitted ItemKNN model from pickle and evaluates it on the test
split using the leave-one-out protocol (1 positive + 99 negatives per user).
Reports HR@k and NDCG@k for k=5,10,20.

Usage:
    python -m src.evaluate --config configs/itemknn.yaml --kcore 20
"""

import argparse
import os
from pathlib import Path

from src.data.dataset import get_n_users_items
from src.metrics.ranking import evaluate_ranking
from src.utils.io import load_config, load_model
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config, kcore_dir):
    """Load ItemKNN model and evaluate on the test set."""
    ckpt_dir = config.get("checkpoint_dir", "results/itemknn")
    model_path = os.path.join(ckpt_dir, "itemknn_model.pkl")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train the model first: python -m src.train --config configs/itemknn.yaml")
        return None

    # load model
    model = load_model(model_path)
    print(f"Loaded ItemKNN from {model_path}")
    print(f"  k_neighbors={model.k}, "
          f"{model.num_users:,} users, {model.num_items:,} items")

    # evaluate
    eval_cfg = config.get("evaluation", {})
    k_values = eval_cfg.get("top_k", [5, 10, 20])
    num_negatives = eval_cfg.get("num_negatives", 99)

    print(f"\nEvaluating on test set (num_negatives={num_negatives})...")
    results = evaluate_ranking(
        model, kcore_dir, split="test",
        k_values=k_values, num_negatives=num_negatives,
    )

    # print results
    print(f"\n{'=' * 55}")
    print(f"  ItemKNN — Test Results")
    print(f"{'=' * 55}")
    for metric, val in sorted(results.items()):
        print(f"    {metric}: {val:.4f}")
    print(f"{'=' * 55}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ItemKNN on HotelRec")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Data config YAML")
    parser.add_argument("--kcore", type=int, default=20,
                        help="K-core value (5 or 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

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

    evaluate(config, kcore_dir)


if __name__ == "__main__":
    main()
