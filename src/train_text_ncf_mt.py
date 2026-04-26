"""
Train Multi-Task TextNCF on HotelRec.

Joint loss: alpha * BPR_loss + (1 - alpha) * MSE_loss on ratings.
The BPR part uses the same triplet sampling as regular TextNCF.
The MSE part uses actual ratings from the dataset.

Usage:
    python -m src.train_text_ncf_mt --config configs/text_ncf_mt.yaml --kcore 20

"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import get_n_users_items, load_split, get_user_positive_items
from src.data.dataset import EvalInteractionDataset
from src.data.text_embeddings import load_text_embeddings, TEXT_EMB_DIR
from src.evaluation.ranking import evaluate_ranking
from src.models.text_ncf_mt import TextNCFMultiTask
from src.utils.io import load_config, save_checkpoint, load_checkpoint
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


# Dataset that returns ratings alongside BPR triplets
class RatedInteractionDataset(Dataset):
    """BPR training dataset that also returns the actual rating for the
    positive item. Negative items get a rating of 0 (unobserved).
    """

    def __init__(self, df: pd.DataFrame, n_items: int, num_negatives: int = 1):
        self.users = df["user_id"].values
        self.items = df["item_id"].values

        # use rating column if available, else default to 1.0
        if "rating" in df.columns:
            self.ratings = df["rating"].values.astype(np.float32)
        else:
            self.ratings = np.ones(len(self.users), dtype=np.float32)

        self.n_items = n_items
        self.num_negatives = num_negatives

        # build user positive set for negative sampling
        self.user_pos = {}
        for u, i in zip(self.users, self.items):
            if u not in self.user_pos:
                self.user_pos[u] = set()
            self.user_pos[u].add(i)

        self._len = len(self.users) * self.num_negatives

    def __len__(self):
        return self._len

    def _sample_neg(self, user: int) -> int:
        pos_set = self.user_pos.get(user, set())
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in pos_set:
                return neg

    def __getitem__(self, idx):
        orig_idx = idx // self.num_negatives
        user = self.users[orig_idx]
        pos_item = self.items[orig_idx]
        neg_item = self._sample_neg(user)
        rating = self.ratings[orig_idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )


def multitask_loss(model, users, pos_items, neg_items, ratings, alpha, device):
    """Combined BPR + MSE loss.

    alpha controls the trade-off:
        loss = alpha * BPR + (1 - alpha) * MSE
    """
    users = users.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    ratings = ratings.to(device)

    # BPR loss on ranking head
    pos_rank, pos_rating_pred = model.forward_both(users, pos_items)
    neg_rank = model(users, neg_items)

    bpr = -torch.log(torch.sigmoid(pos_rank - neg_rank) + 1e-8).mean()

    # MSE loss on rating head
    # normalize ratings to [0, 1] range (original is 1-5)
    ratings_norm = (ratings - 1.0) / 4.0
    mse = torch.nn.functional.mse_loss(pos_rating_pred, ratings_norm)

    loss = alpha * bpr + (1.0 - alpha) * mse
    return loss, bpr.item(), mse.item()


def train_one_epoch(model, loader, optimizer, alpha, device):
    model.train()
    total_loss, total_bpr, total_mse, n = 0.0, 0.0, 0.0, 0

    for users, pos_items, neg_items, ratings in loader:
        loss, bpr_val, mse_val = multitask_loss(
            model, users, pos_items, neg_items, ratings, alpha, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpr += bpr_val
        total_mse += mse_val
        n += 1

    return total_loss / max(n, 1), total_bpr / max(n, 1), total_mse / max(n, 1)


def train(config, kcore_dir, num_users, num_items):
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    paths = config.get("paths", {})
    model_cfg = config.get("model", {})

    alpha = train_cfg.get("alpha", 0.7)

    # device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except RuntimeError:
            print("CUDA available but incompatible -- falling back to CPU")

    # build model
    model = TextNCFMultiTask(
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

    # load text embeddings
    emb_dir = paths.get("text_emb_dir", TEXT_EMB_DIR)
    user_emb, item_emb = load_text_embeddings(emb_dir)
    model.set_text_embeddings(
        torch.from_numpy(user_emb), torch.from_numpy(item_emb))

    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Device: {device}, params: {n_params:,}")
    print(f"  alpha={alpha} (BPR weight), 1-alpha={1-alpha:.2f} (MSE weight)")

    # data
    bs = train_cfg.get("batch_size", 256)
    neg_cfg = config.get("negative_sampling", {})
    eval_cfg = config.get("evaluation", {})

    train_df = load_split(kcore_dir, "train")
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")

    _, n_items = get_n_users_items(kcore_dir)

    # training loader with ratings
    train_ds = RatedInteractionDataset(
        train_df, n_items, num_negatives=neg_cfg.get("num_negatives", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)

    # eval loaders (standard leave-one-out)
    user_pos_all = get_user_positive_items(kcore_dir)
    eval_negs = eval_cfg.get("num_negatives", 99)
    seed = config.get("seed", 42)

    val_ds = EvalInteractionDataset(val_df, n_items, user_pos_all, eval_negs, seed)
    test_ds = EvalInteractionDataset(test_df, n_items, user_pos_all, eval_negs, seed)

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True)

    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

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

    log_dir = paths.get("log_dir", "logs/text_ncf_mt")
    ckpt_dir = paths.get("checkpoint_dir", "results/text_ncf_mt")
    _backup_previous_run(log_dir, ckpt_dir)
    logger = MetricsLogger(log_dir)

    k_values = eval_cfg.get("top_k", [5, 10, 20])
    best_metric = 0.0
    track_key = f"NDCG@{k_values[1] if len(k_values) > 1 else k_values[0]}"

    print(f"\nTraining {epochs} epochs (lr={lr}, bs={bs}, patience={patience})...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        loss, bpr_loss, mse_loss = train_one_epoch(
            model, train_loader, optimizer, alpha, device)

        val_metrics = evaluate_ranking(model, val_loader,
                                       k_values=k_values, device=str(device))
        val_metrics["train_loss"] = loss
        val_metrics["bpr_loss"] = bpr_loss
        val_metrics["mse_loss"] = mse_loss

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
        print(f"  [{epoch:3d}/{epochs}]  loss={loss:.4f} "
              f"(bpr={bpr_loss:.4f} mse={mse_loss:.4f})  "
              f"HR@10={hr10:.4f}  NDCG@10={ndcg10:.4f}  "
              f"lr={cur_lr:.2e}{marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print("=" * 70)
    print(f"Best {track_key}: {best_metric:.4f}")

    # final test evaluation
    print("\nTest evaluation...")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model=model)
        model.to(device)

    test_metrics = evaluate_ranking(model, test_loader,
                                    k_values=k_values, device=str(device))

    print("\n  Test Results:")
    for m, v in sorted(test_metrics.items()):
        print(f"    {m}: {v:.4f}")

    # save
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Saved test metrics to {metrics_path}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task TextNCF")
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
