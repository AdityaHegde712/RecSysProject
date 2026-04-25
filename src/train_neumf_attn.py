"""
Train NeuMF with Attention-Weighted Sub-Ratings (Variant B) on HotelRec.

Architecture: NeuMF (GMF + MLP) extended with a per-user attention mechanism
over 6 hotel sub-rating dimensions (Service, Cleanliness, Location, Value,
Rooms, Sleep Quality). The attention-weighted sub-rating vector is fused into
the final prediction layer, providing interpretable, aspect-aware scoring.

Sub-rating feature engineering
-------------------------------
Item aspect vectors are computed from the **train split only** to avoid
data leakage. Missing sub-rating values are filled with the global mean of
that dimension across train rows that have at least one non-null value.

Uses the same 1-vs-99 eval protocol as every other model in this repo:
score the positive + 99 negatives, rank by score, compute HR@k and NDCG@k.

Writes (same pattern when use_attention=False — see configs/neumf_vanilla.yaml):
    results/<checkpoint_dir>/test_metrics_gmf{g}_mlp{m}.json
    results/<checkpoint_dir>/rating_metrics_gmf{g}_mlp{m}.json
    results/<checkpoint_dir>/best_model_gmf{g}_mlp{m}.pt
    logs/<log_dir>/metrics_gmf{g}_mlp{m}.csv

Usage:
    python -m src.train_neumf_attn --config configs/neumf_attn.yaml    --kcore 20   # enhanced (default)
    python -m src.train_neumf_attn --config configs/neumf_vanilla.yaml --kcore 20   # vanilla ablation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress the "libiomp5md.dll already initialised" OMP warning that appears
# on Windows when conda PyTorch + Intel MKL both register an OpenMP runtime.
# Must be set before any torch or numpy import.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import (
    EvalInteractionDataset,
    InteractionDataset,
    get_n_users_items,
    get_user_positive_items,
    load_split,
)
from src.evaluation.rating import mae_from_predictions, rmse_from_predictions
from src.utils.io import load_checkpoint, load_config, save_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed
from torch.utils.data import DataLoader
from src.models.neumf_attn import N_ASPECTS, NeuMF_Attn

# Sub-rating column names — must match both the parquet schema and the YAML.
SUB_RATING_COLS = ["service", "cleanliness", "location", "value", "rooms", "sleep_quality"]


# ---------------------------------------------------------------------------
# Feature engineering: item aspect vectors (train-split only)
# ---------------------------------------------------------------------------

def build_item_aspects(train_df: pd.DataFrame, n_items: int) -> torch.Tensor:
    """Compute per-hotel mean sub-ratings from the train split.

    Only train rows are used to prevent data leakage.  Missing values are
    filled with the global column mean (computed over non-null train rows).

    Returns a float32 tensor of shape (n_items, N_ASPECTS).
    """
    cols = [c for c in SUB_RATING_COLS if c in train_df.columns]
    missing = [c for c in SUB_RATING_COLS if c not in train_df.columns]
    if missing:
        print(f"  [aspects] Warning: columns not found in parquet — {missing}. "
              f"Setting those dimensions to 0.")

    # Compute per-item mean for available columns.
    if cols:
        item_means = (
            train_df.groupby("item_id")[cols]
            .mean()
            .reindex(range(n_items))       # ensure all item ids are present
        )
    else:
        # No sub-rating columns available at all.
        print("  [aspects] No sub-rating columns found. Aspect vectors will be zero.")
        return torch.zeros(n_items, N_ASPECTS, dtype=torch.float32)

    # Fill global means for items/columns that are entirely NaN.
    global_means = item_means.mean(skipna=True)
    item_means = item_means.fillna(global_means)

    # For any remaining NaN (entire column was NaN), fill with 3.0 (midpoint).
    item_means = item_means.fillna(3.0)

    # Build the full tensor, padding missing columns with zeros.
    aspect_array = np.zeros((n_items, N_ASPECTS), dtype=np.float32)
    for j, col in enumerate(SUB_RATING_COLS):
        if col in cols:
            aspect_array[:, j] = item_means[col].values.astype(np.float32)

    print(f"  [aspects] Built item aspect matrix: {aspect_array.shape}  "
          f"nan-count after fill: {np.isnan(aspect_array).sum()}")
    return torch.tensor(aspect_array, dtype=torch.float32)


# ---------------------------------------------------------------------------
# DataLoader factory (plain interaction-based, no sequences needed)
# ---------------------------------------------------------------------------

def get_neumf_dataloaders(
    kcore_dir: str,
    batch_size: int = 256,
    num_negatives: int = 1,
    eval_negatives: int = 99,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    """Build train/val/test DataLoaders plus metadata."""
    print("[data] Loading split parquets...")
    train_df = load_split(kcore_dir, "train")
    print(f"  train: {len(train_df):,} rows")
    val_df = load_split(kcore_dir, "val")
    print(f"  val  : {len(val_df):,} rows")
    test_df = load_split(kcore_dir, "test")
    print(f"  test : {len(test_df):,} rows")

    print("[data] Reading user/item counts...")
    n_users, n_items = get_n_users_items(kcore_dir)
    print(f"  {n_users:,} users | {n_items:,} items")

    # Build full positive set for negative sampling during eval.
    print("[data] Building user→positive-items map (all splits)...")
    user_pos_all = get_user_positive_items(kcore_dir)
    print(f"  done ({len(user_pos_all):,} users with positives)")

    train_ds = InteractionDataset(train_df, n_items, num_negatives)

    print(f"[data] Sampling {eval_negatives} negatives for val ({len(val_df):,} rows)...")
    val_ds = EvalInteractionDataset(val_df, n_items, user_pos_all, eval_negatives, seed)
    print(f"[data] Sampling {eval_negatives} negatives for test ({len(test_df):,} rows)...")
    test_ds = EvalInteractionDataset(test_df, n_items, user_pos_all, eval_negatives, seed)
    print("[data] Eval datasets ready.")

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "n_users": n_users,
        "n_items": n_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# Ranking evaluator (1-vs-99 protocol)
# ---------------------------------------------------------------------------

def rank_eval(
    model: NeuMF_Attn,
    loader: DataLoader,
    device: str,
    k_values: tuple = (5, 10, 20),
) -> dict[str, float]:
    """Evaluate HR@k and NDCG@k using the 1-vs-99 protocol.

    The EvalInteractionDataset yields (users, items, labels) where
    items[:, 0] is the positive and items[:, 1:] are negatives.
    """
    model.eval()
    metrics: dict[str, float] = {f"HR@{k}": 0.0 for k in k_values}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_values})
    n = 0

    with torch.no_grad():
        for users, items, _labels in loader:
            users = users.to(device)       # (B,)
            items = items.to(device)       # (B, C)  C = 1 + eval_negatives

            scores = model.score_candidates(users, items)  # (B, C)
            _, order = torch.sort(scores, dim=1, descending=True)
            # Positive is always at column 0; find its rank after sorting.
            ranks = (order == 0).float().argmax(dim=1).cpu().numpy()

            n += len(ranks)
            for k in k_values:
                hit = (ranks < k).astype(np.float32)
                metrics[f"HR@{k}"] += float(hit.sum())
                metrics[f"NDCG@{k}"] += float(
                    np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0).sum()
                )

    return {key: v / max(n, 1) for key, v in metrics.items()}


def val_bpr_loss(
    model: NeuMF_Attn,
    loader: DataLoader,
    device: str,
) -> float:
    """Compute mean BPR loss on the validation set (no gradient).

    The EvalInteractionDataset yields (users, items, labels) where
    items[:, 0] is the positive and items[:, 1:] are negatives.  We use
    items[:, 1] as a single representative negative to mirror the BPR
    training objective without any extra negative sampling.
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for users, items, _labels in loader:
            users = users.to(device)   # (B,)
            items = items.to(device)   # (B, C)
            pos = items[:, 0]          # (B,)  — positive item
            neg = items[:, 1]          # (B,)  — first sampled negative
            pos_s, neg_s = model(users, pos, neg)
            loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
            total += loss.item()
            n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Score → rating calibration (mirrors calibrate_sasrec in train_sasrec.py)
# ---------------------------------------------------------------------------

def calibrate_neumf(
    model: NeuMF_Attn,
    device: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Fit a linear score→rating calibration on val, evaluate on test.

    Robust to near-constant score distributions: falls back to train-mean
    predictor if lstsq fails or scores have near-zero variance.
    """
    def score_fn(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        out = np.empty(len(users), dtype=np.float32)
        BS = 2048
        model.eval()
        with torch.no_grad():
            for start in range(0, len(users), BS):
                end = min(start + BS, len(users))
                u = torch.tensor(users[start:end], dtype=torch.long, device=device)
                i = torch.tensor(items[start:end], dtype=torch.long, device=device)
                # score_candidates expects (B, C); wrap item as (B, 1).
                scores = model.score_candidates(u, i.unsqueeze(1)).squeeze(1)
                out[start:end] = scores.cpu().numpy()
        return out

    vu = val_df["user_id"].astype(np.int64).values
    vi = val_df["item_id"].astype(np.int64).values
    vy = val_df["rating"].astype(np.float32).values
    vs = np.nan_to_num(score_fn(vu, vi).astype(np.float64),
                       nan=0.0, posinf=0.0, neginf=0.0)

    a, b, note = 0.0, float(vy.mean()), None
    vs_std = float(vs.std())
    if vs_std < 1e-10 or not np.isfinite(vs_std):
        note = f"scores near-constant (std={vs_std:.3e})"
    else:
        vs_n = (vs - vs.mean()) / vs_std
        A = np.vstack([vs_n, np.ones_like(vs_n)]).T
        try:
            coef, *_ = np.linalg.lstsq(A, vy.astype(np.float64), rcond=None)
            a = float(coef[0]) / vs_std
            b = float(coef[1]) - float(coef[0]) * vs.mean() / vs_std
        except np.linalg.LinAlgError as e:
            note = f"lstsq failed: {e}"

    tu = test_df["user_id"].astype(np.int64).values
    ti = test_df["item_id"].astype(np.int64).values
    ty = test_df["rating"].astype(np.float32).values
    ts = np.nan_to_num(score_fn(tu, ti).astype(np.float64),
                       nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(a * ts + b, 1.0, 5.0).astype(np.float32)

    out = {
        "rmse_calibrated": rmse_from_predictions(pred, ty),
        "mae_calibrated":  mae_from_predictions(pred, ty),
        "calibration_a":   float(a),
        "calibration_b":   float(b),
        "n":               int(len(ty)),
        "val_score_std":   vs_std,
    }
    if note:
        out["fallback_note"] = note
    return out


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            dev = torch.device("cuda")
        except RuntimeError:
            pass
    return dev


# ---------------------------------------------------------------------------
# BPR training step
# ---------------------------------------------------------------------------

def bpr_step(
    model: NeuMF_Attn,
    users: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    pos_s, neg_s = model(users, pos, neg)
    loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
    return loss, loss.item()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(config: dict, kcore_dir: str):
    model_cfg  = config["model"]
    train_cfg  = config.get("training", {})
    sched_cfg  = config.get("scheduler", {})
    eval_cfg   = config.get("evaluation", {})
    neg_cfg    = config.get("negative_sampling", {})
    paths      = config.get("paths", {})

    device = pick_device()

    # ---- Hyper-parameters --------------------------------------------------
    gmf_dim       = int(model_cfg.get("gmf_dim", 64))
    mlp_dim       = int(model_cfg.get("mlp_dim", 64))
    mlp_layers    = list(model_cfg.get("mlp_layers", [256, 128, 64]))
    dropout       = float(model_cfg.get("dropout", 0.2))
    use_attention = bool(model_cfg.get("use_attention", True))

    # ---- DataLoaders -------------------------------------------------------
    loaders = get_neumf_dataloaders(
        kcore_dir,
        batch_size=int(train_cfg.get("batch_size", 256)),
        num_negatives=int(neg_cfg.get("num_negatives", 1)),
        eval_negatives=int(eval_cfg.get("num_negatives", 99)),
        seed=42,
    )
    n_users   = loaders["n_users"]
    n_items   = loaders["n_items"]
    train_df  = loaders["train_df"]

    print(f"Dataset: {n_users:,} users, {n_items:,} items, "
          f"train batches={len(loaders['train'])}")

    # ---- Sub-rating aspect vectors (train-only, no leakage) ---------------
    # Skipped when use_attention is False — the vanilla ablation doesn't
    # touch the aspect columns at all.
    if use_attention:
        print("Building item aspect vectors from train split...")
        item_aspects = build_item_aspects(train_df, n_items).to(device)
    else:
        print("use_attention=False — vanilla NeuMF ablation (no aspect matrix).")
        item_aspects = None

    # ---- Model -------------------------------------------------------------
    model = NeuMF_Attn(
        n_users=n_users,
        n_items=n_items,
        gmf_dim=gmf_dim,
        mlp_dim=mlp_dim,
        mlp_layers=mlp_layers,
        dropout=dropout,
        item_aspects=item_aspects,
        use_attention=use_attention,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | gmf_dim={gmf_dim} | mlp_dim={mlp_dim} "
          f"| mlp_layers={mlp_layers} | use_attention={use_attention} "
          f"| params={n_params:,}")

    # ---- Optimiser & scheduler --------------------------------------------
    lr  = float(train_cfg.get("lr", 1e-3))
    wd  = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 30))
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-5)),
        )

    # ---- Logging & checkpointing ------------------------------------------
    k_values  = eval_cfg.get("top_k", [5, 10, 20])
    track_key = f"HR@{k_values[1] if len(k_values) > 1 else k_values[0]}"
    patience  = int(train_cfg.get("patience", epochs))

    log_dir  = paths.get("log_dir", "logs/neumf_attn")
    ckpt_dir = paths.get("checkpoint_dir", "results/neumf_attn")
    run_suffix = f"gmf{gmf_dim}_mlp{mlp_dim}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    logger    = MetricsLogger(log_dir, filename=f"metrics_{run_suffix}.csv")
    best_path = os.path.join(ckpt_dir, f"best_model_{run_suffix}.pt")

    best_metric = 0.0
    best_epoch  = 0
    patience_c  = 0

    print(f"\nTraining {epochs} epochs (lr={lr}, bs={train_cfg.get('batch_size', 256)})...")
    print("=" * 72)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        ep_t = time.time()

        train_iter = tqdm(
            loaders["train"],
            desc=f"  Epoch {epoch:3d}/{epochs} [train]",
            leave=False,
            unit="batch",
        )
        for users, pos_items, neg_items in train_iter:
            users     = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            loss, loss_val = bpr_step(model, users, pos_items, neg_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss_val
            n_batches  += 1
            train_iter.set_postfix(loss=f"{loss_val:.4f}")

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        val_m = rank_eval(model, loaders["val"], str(device), tuple(k_values))
        val_m["val_loss"]      = val_bpr_loss(model, loaders["val"], str(device))
        val_m["train_loss"]    = total_loss / max(n_batches, 1)
        val_m["lr"]            = cur_lr
        val_m["epoch_time_s"]  = round(time.time() - ep_t, 2)
        logger.log(epoch, val_m)

        cur    = val_m[track_key]
        marker = ""
        if cur > best_metric:
            best_metric = cur
            best_epoch  = epoch
            patience_c  = 0
            save_checkpoint(model, optimizer, epoch, best_path)
            marker = " *"
        else:
            patience_c += 1

        show = ("HR@5", "HR@10", "HR@20", "NDCG@10", "train_loss", "val_loss")
        metric_str = "  ".join(
            f"{k}: {val_m[k]:.4f}" for k in show if k in val_m
        )
        print(f"  [{epoch:3d}/{epochs}]  {metric_str}  "
              f"lr={cur_lr:.1e}  t={val_m['epoch_time_s']}s{marker}")

        if patience_c >= patience:
            print(f"Early stop at epoch {epoch} (no gain for {patience} epochs)")
            break

    # ---- Reload best checkpoint, eval on test ------------------------------
    if os.path.exists(best_path):
        model, _ = load_checkpoint(best_path, model)
        model = model.to(device)
    model.eval()

    test_m = rank_eval(model, loaders["test"], str(device), tuple(k_values))
    test_m[f"best_val_{track_key}"] = best_metric
    test_m["best_epoch"]   = best_epoch
    test_m["gmf_dim"]      = gmf_dim
    test_m["mlp_dim"]      = mlp_dim
    test_m["mlp_layers"]   = mlp_layers
    test_m["use_attention"] = use_attention
    test_m["total_train_time_s"] = round(time.time() - t0, 2)

    print("\n" + "=" * 72)
    print("NeuMF_Attn test results:")
    for k in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
        if k in test_m:
            print(f"  {k}: {test_m[k]:.4f}")
    print(f"  best val {track_key}: {best_metric:.4f} @ epoch {best_epoch}")
    print(f"  total time: {test_m['total_train_time_s']}s")

    test_metrics_path = os.path.join(ckpt_dir, f"test_metrics_{run_suffix}.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_m, f, indent=2)

    # ---- Calibrated rating metrics ----------------------------------------
    val_df  = loaders["val_df"]
    test_df = loaders["test_df"]
    rating_m = calibrate_neumf(model, str(device), val_df, test_df)
    rating_m["gmf_dim"]    = gmf_dim
    rating_m["mlp_dim"]    = mlp_dim
    rating_m["mlp_layers"] = mlp_layers
    rating_m["use_attention"] = use_attention

    rating_metrics_path = os.path.join(ckpt_dir, f"rating_metrics_{run_suffix}.json")
    with open(rating_metrics_path, "w") as f:
        json.dump(rating_m, f, indent=2)

    print(f"NeuMF_Attn calibrated: RMSE={rating_m['rmse_calibrated']:.4f}  "
          f"MAE={rating_m['mae_calibrated']:.4f}  "
          f"(a={rating_m['calibration_a']:.4f}, b={rating_m['calibration_b']:.4f})")

    print("\nSaved:")
    print(f"  {best_path}")
    print(f"  {test_metrics_path}")
    print(f"  {rating_metrics_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train NeuMF with Attention-Weighted Sub-Ratings (Variant B)"
    )
    parser.add_argument("--config", default="configs/neumf_attn.yaml",
                        help="Path to YAML config (default: configs/neumf_attn.yaml)")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Path to data YAML config")
    parser.add_argument("--kcore", type=int, default=20,
                        help="k-core threshold used during preprocessing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config   = load_config(args.config)
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
            f"Run:\n"
            f"  python -m src.data.preprocess --kcore {args.kcore}\n"
            f"  python -m src.data.split --kcore {args.kcore}"
        )

    train(config, kcore_dir)


if __name__ == "__main__":
    main()
