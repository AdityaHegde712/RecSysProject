"""
Train LightGCN (He et al., SIGIR 2020) on HotelRec.

Usage:
    python -m src.train_lightgcn --config configs/lightgcn.yaml --kcore 20
    python -m src.train_lightgcn --config configs/lightgcn.yaml --kcore 20 --num-layers 2
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import get_dataloaders, get_n_users_items, load_split
# Silence tqdm progress bars in evaluate_ranking (they flood training logs).
from functools import partial
import src.evaluation.ranking as _rk
_rk.tqdm = partial(_rk.tqdm, disable=True)
from src.evaluation.ranking import evaluate_ranking
from src.models.lightgcn import LightGCN, build_norm_adj
from src.utils.io import load_config, save_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


def bpr_step(model, users, pos_items, neg_items, device, reg: float):
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)

    pos_scores, neg_scores, (u0, pi0, ni0) = model.score_triplet(
        users, pos_items, neg_items
    )
    bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    # L2 reg on layer-0 embeddings (as in the LightGCN paper).
    reg_loss = (u0.pow(2).sum() + pi0.pow(2).sum() + ni0.pow(2).sum()) / (
        2 * users.size(0)
    )
    return bpr + reg * reg_loss, bpr.item()


def train_one_epoch(model, loader, optimizer, device, reg: float):
    model.train()  # invalidates cache
    total_loss, total_bpr, n = 0.0, 0.0, 0
    for users, pos_items, neg_items in loader:
        loss, bpr_val = bpr_step(model, users, pos_items, neg_items, device, reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_bpr += bpr_val
        n += 1
    return total_loss / max(n, 1), total_bpr / max(n, 1)


def pick_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            print("CUDA available but incompatible -- falling back to CPU")
    return device


def train(config, kcore_dir: str, num_users: int, num_items: int, layers_override=None):
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    paths = config.get("paths", {})
    model_cfg = config["model"]

    device = pick_device()

    # Build normalized adjacency from training interactions only.
    train_df = load_split(kcore_dir, "train")
    print(f"Building normalized adjacency from {len(train_df):,} train edges...")
    adj_hat = build_norm_adj(
        train_df["user_id"].values.astype(np.int64),
        train_df["item_id"].values.astype(np.int64),
        num_users,
        num_items,
    ).to(device)

    num_layers = int(layers_override) if layers_override is not None else int(
        model_cfg.get("num_layers", 3)
    )
    embed_dim = int(model_cfg.get("embedding_dim", model_cfg.get("embed_dim", 64)))

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=embed_dim,
        num_layers=num_layers,
        adj_hat=adj_hat,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Layers: {num_layers} | dim: {embed_dim} | "
          f"params: {n_params:,}")

    bs = int(train_cfg.get("batch_size", 2048))
    neg_cfg = config.get("negative_sampling", {})
    eval_cfg = config.get("evaluation", {})

    loaders = get_dataloaders(
        kcore_dir,
        batch_size=bs,
        num_negatives=int(neg_cfg.get("num_negatives", 1)),
        eval_negatives=int(eval_cfg.get("num_negatives", 99)),
        seed=config.get("split", {}).get("seed", 42),
    )
    print(f"Train: {len(loaders['train'].dataset):,}  "
          f"Val: {len(loaders['val'].dataset):,}  "
          f"Test: {len(loaders['test'].dataset):,}")

    lr = float(train_cfg.get("lr", train_cfg.get("learning_rate", 1e-3)))
    wd = float(train_cfg.get("weight_decay", 0.0))
    reg = float(train_cfg.get("bpr_reg", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 30))
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )

    patience = int(train_cfg.get("patience", epochs))
    patience_counter = 0

    log_dir = paths.get("log_dir", "logs/lightgcn")
    ckpt_dir = paths.get("checkpoint_dir", "results/lightgcn")
    logger = MetricsLogger(log_dir, filename=f"metrics_L{num_layers}.csv")

    k_values = eval_cfg.get("top_k", [5, 10, 20])
    track_key = f"HR@{k_values[1] if len(k_values) > 1 else k_values[0]}"
    best_metric = 0.0
    best_epoch = 0

    print(f"\nTraining {epochs} epochs (lr={lr}, bs={bs}, reg={reg})...")
    print("=" * 72)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        loss, bpr_val = train_one_epoch(model, loaders["train"], optimizer, device, reg)

        # Prime cache for fast eval.
        model.eval()
        model.cache_embeddings()
        val_metrics = evaluate_ranking(
            model, loaders["val"], k_values=k_values, device=str(device)
        )
        val_metrics["train_loss"] = loss
        val_metrics["bpr_loss"] = bpr_val

        cur_lr = optimizer.param_groups[0]["lr"]
        val_metrics["lr"] = cur_lr
        val_metrics["epoch_time_s"] = round(time.time() - ep_start, 2)
        if scheduler:
            scheduler.step()

        logger.log(epoch, val_metrics)

        cur_metric = val_metrics[track_key]
        improved = cur_metric > best_metric
        marker = ""
        if improved:
            best_metric = cur_metric
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(ckpt_dir, f"best_model_L{num_layers}.pt"),
            )
            marker = " *"
        else:
            patience_counter += 1

        show = ("HR@5", "HR@10", "HR@20", "NDCG@10", "train_loss")
        metric_str = "  ".join(f"{k}: {val_metrics[k]:.4f}" for k in show if k in val_metrics)
        print(f"  [{epoch:3d}/{epochs}]  {metric_str}  "
              f"lr={cur_lr:.1e}  t={val_metrics['epoch_time_s']}s{marker}")

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch} (no gain for {patience} epochs)")
            break

    # Final test eval using best checkpoint.
    best_path = os.path.join(ckpt_dir, f"best_model_L{num_layers}.pt")
    if os.path.exists(best_path):
        from src.utils.io import load_checkpoint
        model, _ = load_checkpoint(best_path, model)
        model = model.to(device)
    model.eval()
    model.cache_embeddings()
    test_metrics = evaluate_ranking(
        model, loaders["test"], k_values=k_values, device=str(device)
    )
    test_metrics["best_val_" + track_key] = best_metric
    test_metrics["best_epoch"] = best_epoch
    test_metrics["num_layers"] = num_layers
    test_metrics["embed_dim"] = embed_dim
    test_metrics["total_train_time_s"] = round(time.time() - t0, 2)

    print("\n" + "=" * 72)
    print(f"LightGCN (L={num_layers}) test results:")
    for k in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
        if k in test_metrics:
            print(f"  {k}: {test_metrics[k]:.4f}")
    print(f"  best val {track_key}: {best_metric:.4f} @ epoch {best_epoch}")
    print(f"  total time: {test_metrics['total_train_time_s']}s")

    # Save test metrics as JSON.
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(ckpt_dir, f"test_metrics_L{num_layers}.json")
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  saved: {out_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGCN on HotelRec")
    parser.add_argument("--config", default="configs/lightgcn.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override config num_layers (useful for sweeps)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = (load_config(args.data_config)
                if Path(args.data_config).exists() else {})

    set_seed(args.seed)

    kcore_dir = os.path.join(
        data_cfg.get("dataset", {}).get("processed_dir", "data/processed"),
        f"{args.kcore}core",
    )
    if not os.path.isdir(kcore_dir):
        raise FileNotFoundError(
            f"Processed data not found: {kcore_dir}\n"
            f"Run: python -m src.data.preprocess --kcore {args.kcore}"
        )

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: kcore={args.kcore}, {num_users:,} users, {num_items:,} items")

    train(config, kcore_dir, num_users, num_items, layers_override=args.num_layers)


if __name__ == "__main__":
    main()
