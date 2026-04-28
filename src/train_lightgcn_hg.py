"""
Train LightGCN-HG (metadata-augmented LightGCN) on HotelRec.

Same BPR / cosine-LR / early-stop loop as train_lightgcn.py; the only change is the graph (build_hg_norm_adj) and the model (LightGCNHG). Tiers are
configurable so that the vanilla bipartite model, a g_id-only variant, and the full g_id+region+country variant can all be run from this one entry point.

Usage
-----
    # Full three-tier
    python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20

    # Ablation: only g_id pivots
    python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20 \
        --tiers g_id

    # Ablation: no tiers (reproduces vanilla LightGCN under this harness)
    python -m src.train_lightgcn_hg --config configs/lightgcn_hg.yaml --kcore 20 \
        --tiers none
"""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import get_dataloaders, get_n_users_items, load_split
import src.evaluation.ranking as _rk
_rk.tqdm = partial(_rk.tqdm, disable=True)
from src.evaluation.ranking import evaluate_ranking
from src.models.lightgcn_hg import LightGCNHG, build_hg_norm_adj
from src.utils.io import load_config, save_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


TIER_SHORT = {"g_id": "g", "region_slug": "r", "country_slug": "c"}


def tier_suffix(tiers: list[str]) -> str:
    if not tiers:
        return "none"
    return "".join(TIER_SHORT[t] for t in tiers)


def parse_tiers_arg(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if raw.strip().lower() == "none":
        return []
    parsed = [t.strip() for t in raw.split(",") if t.strip()]
    valid = set(TIER_SHORT.keys())
    for t in parsed:
        if t not in valid:
            raise ValueError(
                f"Unknown tier '{t}'. Valid: {sorted(valid)} or 'none'."
            )
    return parsed


def bpr_step(model, users, pos_items, neg_items, device, reg: float):
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)

    pos_scores, neg_scores, (u0, pi0, ni0) = model.score_triplet(
        users, pos_items, neg_items
    )
    bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    reg_loss = (u0.pow(2).sum() + pi0.pow(2).sum() + ni0.pow(2).sum()) / (
        2 * users.size(0)
    )
    return bpr + reg * reg_loss, bpr.item()


def train_one_epoch(model, loader, optimizer, device, reg: float):
    model.train()
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


def load_hotel_meta(path: str, tiers: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} -- run: python -m scripts.extract_hotel_meta --kcore 20"
        )
    df = pd.read_parquet(path)
    needed = ["item_id"] + tiers
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"hotel_meta missing column '{c}'")
    return df[needed].copy()


def train(config, kcore_dir: str, num_users: int, num_items: int,
          tiers: list[str], meta_path: str,
          layers_override=None, dim_override=None):
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    paths = config.get("paths", {})
    model_cfg = config["model"]

    device = pick_device()

    # Build hetero adjacency from training interactions + hotel metadata.
    train_df = load_split(kcore_dir, "train")
    hotel_meta = load_hotel_meta(meta_path, tiers) if tiers else pd.DataFrame(
        {"item_id": np.arange(num_items)}
    )
    print(f"Building heterogeneous adjacency from "
          f"{len(train_df):,} train edges + {len(hotel_meta):,} hotels "
          f"(tiers: {tiers or 'none'})...")
    adj_hat, graph_meta = build_hg_norm_adj(
        train_df["user_id"].values.astype(np.int64),
        train_df["item_id"].values.astype(np.int64),
        hotel_meta if tiers else pd.DataFrame(columns=["item_id"]),
        num_users,
        num_items,
        tiers=tiers,
    )
    adj_hat = adj_hat.to(device)
    for k, v in graph_meta["tier_sizes"].items():
        print(f"    tier '{k}': {v:,} nodes")
    print(f"    total nodes: {graph_meta['n_total']:,}  "
          f"edges (directed): {graph_meta['n_edges_sym']:,}")

    num_layers = int(layers_override) if layers_override is not None else int(
        model_cfg.get("num_layers", 1)
    )
    embed_dim = int(dim_override) if dim_override is not None else int(
        model_cfg.get("embedding_dim", model_cfg.get("embed_dim", 256))
    )

    model = LightGCNHG(
        num_users=num_users,
        num_items=num_items,
        graph_meta=graph_meta,
        embed_dim=embed_dim,
        num_layers=num_layers,
        adj_hat=adj_hat,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | Layers: {num_layers} | dim: {embed_dim} | "
          f"params: {n_params:,}")

    bs = int(train_cfg.get("batch_size", 8192))
    neg_cfg = config.get("negative_sampling", {})
    eval_cfg = config.get("evaluation", {})

    loaders = get_dataloaders(
        kcore_dir,
        batch_size=bs,
        num_negatives=int(neg_cfg.get("num_negatives", 2)),
        eval_negatives=int(eval_cfg.get("num_negatives", 99)),
        seed=config.get("split", {}).get("seed", 42),
    )
    print(f"Train: {len(loaders['train'].dataset):,}  "
          f"Val: {len(loaders['val'].dataset):,}  "
          f"Test: {len(loaders['test'].dataset):,}")

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    reg = float(train_cfg.get("bpr_reg", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 80))
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-5)),
        )

    patience = int(train_cfg.get("patience", epochs))
    patience_counter = 0

    log_dir = paths.get("log_dir", "logs/lightgcn_hg")
    ckpt_dir = paths.get("checkpoint_dir", "results/lightgcn_hg")
    tsuf = tier_suffix(tiers)
    run_suffix = f"L{num_layers}_d{embed_dim}_{tsuf}"
    logger = MetricsLogger(log_dir, filename=f"metrics_{run_suffix}.csv")

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
                os.path.join(ckpt_dir, f"best_model_{run_suffix}.pt"),
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
    best_path = os.path.join(ckpt_dir, f"best_model_{run_suffix}.pt")
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
    test_metrics["tiers"] = tiers
    test_metrics["n_total_nodes"] = int(graph_meta["n_total"])
    test_metrics["n_edges_directed"] = int(graph_meta["n_edges_sym"])
    test_metrics["total_train_time_s"] = round(time.time() - t0, 2)

    print("\n" + "=" * 72)
    print(f"LightGCN-HG (L={num_layers}, tiers={tiers}) test results:")
    for k in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
        if k in test_metrics:
            print(f"  {k}: {test_metrics[k]:.4f}")
    print(f"  best val {track_key}: {best_metric:.4f} @ epoch {best_epoch}")
    print(f"  total time: {test_metrics['total_train_time_s']}s")

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(ckpt_dir, f"test_metrics_{run_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  saved: {out_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LightGCN-HG on HotelRec")
    parser.add_argument("--config", default="configs/lightgcn_hg.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--tiers", type=str, default=None,
                        help="Comma-separated tiers or 'none'. "
                             "Valid: g_id,region_slug,country_slug")
    parser.add_argument("--meta-path", type=str,
                        default="data/processed/hotel_meta/hotel_meta.parquet")
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

    default_tiers = config.get("graph", {}).get(
        "tiers", ["g_id", "region_slug", "country_slug"]
    )
    tiers = parse_tiers_arg(args.tiers, default_tiers)

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: kcore={args.kcore}, {num_users:,} users, {num_items:,} items")
    print(f"Tiers:   {tiers or 'none (bipartite only)'}")

    train(
        config, kcore_dir, num_users, num_items,
        tiers=tiers, meta_path=args.meta_path,
        layers_override=args.num_layers,
        dim_override=args.embed_dim,
    )


if __name__ == "__main__":
    main()
