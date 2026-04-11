"""
Training entry point for HotelRec — GMF baseline.

Trains a Generalized Matrix Factorization model (He et al., 2017) on the
implicit recommendation task using BPR loss. The raw dataset is ~50GB, so
we stream through DataLoaders and never hold the full dataset in RAM.

Usage:
    python -m src.train --config configs/gmf.yaml --kcore 20
"""

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.dataset import get_dataloaders, get_n_users_items
from src.metrics.ranking import evaluate_ranking
from src.models.common import build_model
from src.utils.io import load_config, save_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def bpr_loss(model, users, pos_items, neg_items, device):
    """BPR pairwise loss: -mean(log(sigmoid(pos - neg)))."""
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)

    pos_scores = model(users, pos_items)
    neg_scores = model(users, neg_items)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def train_one_epoch(model, loader, optimizer, device):
    """One epoch of BPR training."""
    model.train()
    total_loss, n = 0.0, 0

    for users, pos_items, neg_items in loader:
        loss = bpr_loss(model, users, pos_items, neg_items, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config, kcore_dir, num_users, num_items):
    """Full training loop for GMF."""
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    paths = config.get("paths", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, num_users, num_items).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, params: {n_params:,}")

    # data
    bs = train_cfg.get("batch_size", 256)
    neg_cfg = config.get("negative_sampling", {})
    eval_cfg = config.get("evaluation", {})

    loaders = get_dataloaders(
        kcore_dir, batch_size=bs,
        num_negatives=neg_cfg.get("num_negatives", 4),
        eval_negatives=eval_cfg.get("num_negatives", 99),
        seed=config.get("split", {}).get("seed", 42),
    )

    print(f"Train: {len(loaders['train'].dataset):,}, "
          f"Val: {len(loaders['val'].dataset):,}")

    # optimizer
    lr = train_cfg.get("lr", train_cfg.get("learning_rate", 1e-3))
    wd = train_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # cosine scheduler
    epochs = train_cfg.get("epochs", 20)
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=sched_cfg.get("min_lr", 1e-6))

    # early stopping
    patience = train_cfg.get("patience", epochs)
    patience_counter = 0

    # logging
    log_dir = paths.get("log_dir", "logs/gmf")
    ckpt_dir = paths.get("checkpoint_dir", "results/gmf")
    logger = MetricsLogger(log_dir)

    k_values = eval_cfg.get("top_k", [5, 10, 20])
    best_metric = 0.0
    track_key = f"HR@{k_values[1] if len(k_values) > 1 else k_values[0]}"

    print(f"\nTraining {epochs} epochs (lr={lr}, bs={bs})...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, device)

        val_metrics = evaluate_ranking(model, loaders["val"],
                                       k_values=k_values, device=str(device))
        val_metrics["train_loss"] = loss
        cur_metric = val_metrics[track_key]

        cur_lr = optimizer.param_groups[0]["lr"]
        val_metrics["lr"] = cur_lr
        if scheduler:
            scheduler.step()

        logger.log(epoch, val_metrics)

        # check improvement
        improved = cur_metric > best_metric
        marker = ""
        if improved:
            best_metric = cur_metric
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(ckpt_dir, "best_model.pt"))
            marker = " *"
        else:
            patience_counter += 1

        metric_str = "  ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()
                               if k != "lr")
        print(f"  [{epoch:3d}/{epochs}]  {metric_str}  lr={cur_lr:.1e}{marker}")

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch} (no gain for {patience} epochs)")
            break

    save_checkpoint(model, optimizer, epoch, os.path.join(ckpt_dir, "last_model.pt"))
    print(f"\nBest {track_key}: {best_metric:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_dir}")

    return best_metric


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GMF on HotelRec")
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
