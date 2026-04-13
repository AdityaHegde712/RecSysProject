"""
Training entry point for HotelRec — ItemKNN baseline.

ItemKNN doesn't have a training loop — it just fits on the interaction
matrix (builds item-item cosine similarity). No GPU needed.

Usage:
    python -m src.train --config configs/itemknn.yaml --kcore 20
"""

import argparse
import os
import time
from pathlib import Path

from src.data.dataset import load_split, get_n_users_items
from src.metrics.ranking import evaluate_ranking
from src.models.common import build_model
from src.utils.io import load_config, save_model, load_model
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config, kcore_dir, num_users, num_items):
    """Fit ItemKNN and evaluate on validation set."""
    eval_cfg = config.get("evaluation", {})
    ckpt_dir = config.get("checkpoint_dir", "results/itemknn")
    log_dir = config.get("log_dir", "results/logs/itemknn")

    # build model
    model = build_model(config, num_users, num_items)

    # load training data
    train_df = load_split(kcore_dir, "train")
    print(f"Train interactions: {len(train_df):,}")

    # fit — this is the only "training" step for ItemKNN
    print("\nFitting ItemKNN (building similarity matrix)...")
    t0 = time.time()
    model.fit(train_df, num_users, num_items)
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.1f}s")

    # save model
    model_path = os.path.join(ckpt_dir, "itemknn_model.pkl")
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # evaluate on validation set
    k_values = eval_cfg.get("top_k", [5, 10, 20])
    num_negatives = eval_cfg.get("num_negatives", 99)

    print(f"\nEvaluating on validation set (num_negatives={num_negatives})...")
    val_df = load_split(kcore_dir, "val")
    val_results = evaluate_ranking(
        model, kcore_dir, split="val",
        k_values=k_values, num_negatives=num_negatives,
    )

    # log results
    logger = MetricsLogger(log_dir)
    logger.log(1, {**val_results, "fit_time_s": fit_time})

    # print results
    print(f"\n{'=' * 55}")
    print(f"  ItemKNN — Validation Results")
    print(f"{'=' * 55}")
    for metric, val in sorted(val_results.items()):
        print(f"    {metric}: {val:.4f}")
    print(f"{'=' * 55}")

    return val_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ItemKNN on HotelRec")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Data config YAML")
    parser.add_argument("--kcore", type=int, default=20,
                        help="K-core value (5 or 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = (load_config(args.data_config)
                if Path(args.data_config).exists() else {})

    set_seed(args.seed)

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

    train(config, kcore_dir, num_users, num_items)


if __name__ == "__main__":
    main()
