"""
Train TextNCF on HotelRec.

Usage:
    python -m src.train_text_ncf --config configs/text_ncf.yaml --kcore 20
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataset import get_dataloaders, get_n_users_items
from src.data.text_embeddings import load_text_embeddings, TEXT_EMB_DIR
from src.evaluation.ranking import evaluate_ranking
from src.models.text_ncf import TextNCF
from src.utils.io import load_config, save_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


def _backup_previous_run(log_dir, ckpt_dir):
    """Move existing metrics/checkpoints to timestamped backups so re-runs
    don't append to stale CSV files or silently overwrite results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [
        os.path.join(log_dir, "metrics.csv"),
        os.path.join(ckpt_dir, "test_metrics.json"),
        os.path.join(ckpt_dir, "best_model.pt"),
    ]:
        if os.path.exists(path):
            backup = f"{path}.{ts}.bak"
            shutil.move(path, backup)
            print(f"  backed up {path} → {backup}")


def bpr_loss(model, users, pos_items, neg_items, device):
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)

    pos_scores = model(users, pos_items)
    neg_scores = model(users, neg_items)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def train_one_epoch(model, loader, optimizer, device):
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


def train(config, kcore_dir, num_users, num_items):
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})

    # device selection
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            print("CUDA available but incompatible -- falling back to CPU")

    # build model
    model = TextNCF(
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

    # load text embeddings into model buffers
    emb_dir = paths.get("text_emb_dir", TEXT_EMB_DIR)
    user_emb, item_emb = load_text_embeddings(emb_dir)
    model.set_text_embeddings(
        torch.from_numpy(user_emb),
        torch.from_numpy(item_emb),
    )

    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Device: {device}, params: {n_params:,}")
    print(f"  gmf={'on' if model.use_gmf else 'off'}, "
          f"text={'on' if model.use_text else 'off'}")

    # data loaders (shared)
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

    lr = train_cfg.get("lr", train_cfg.get("learning_rate", 1e-3))
    wd = train_cfg.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = train_cfg.get("epochs", 30)
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=sched_cfg.get("min_lr", 1e-6))

    patience = train_cfg.get("patience", epochs)
    patience_counter = 0

    log_dir = paths.get("log_dir", "logs/text_ncf")
    ckpt_dir = paths.get("checkpoint_dir", "results/text_ncf")
    _backup_previous_run(log_dir, ckpt_dir)
    logger = MetricsLogger(log_dir)

    k_values = eval_cfg.get("top_k", [5, 10, 20])
    best_metric = 0.0
    track_key = f"NDCG@{k_values[1] if len(k_values) > 1 else k_values[0]}"

    print(f"\nTraining {epochs} epochs (lr={lr}, bs={bs}, patience={patience})...")
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

        hr10 = val_metrics.get("HR@10", 0)
        ndcg10 = val_metrics.get("NDCG@10", 0)
        print(f"  [{epoch:3d}/{epochs}]  loss={loss:.4f}  "
              f"HR@10={hr10:.4f}  NDCG@10={ndcg10:.4f}  "
              f"lr={cur_lr:.2e}{marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print("=" * 65)
    print(f"Best {track_key}: {best_metric:.4f}")

    # final test evaluation
    print("\nTest evaluation...")
    from src.utils.io import load_checkpoint as _load_ckpt
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_path):
        _load_ckpt(best_path, model=model)
        model.to(device)

    test_metrics = evaluate_ranking(model, loaders["test"],
                                    k_values=k_values, device=str(device))

    print("\n  Test Results:")
    for m, v in sorted(test_metrics.items()):
        print(f"    {m}: {v:.4f}")

    # save test metrics as JSON
    results_dir = ckpt_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Saved test metrics to {metrics_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TextNCF")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config (useful for smoke tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
        config["training"]["patience"] = min(
            config["training"].get("patience", args.epochs), args.epochs)
        print(f"[override] epochs={args.epochs}")

    set_seed(config.get("seed", 42))

    kcore_dir = os.path.join("data", "processed", f"{args.kcore}core")
    if not os.path.isdir(kcore_dir):
        print(f"No data at {kcore_dir}. Run preprocessing first.")
        return

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: {num_users:,} users, {num_items:,} items")

    train(config, kcore_dir, num_users, num_items)


if __name__ == "__main__":
    main()
