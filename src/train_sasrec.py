"""
Train SASRec (Kang & McAuley, ICDM 2018) on HotelRec.

Uses the sequential dataloader from ``src/data/sequential.py`` and the SASRec model from ``src/models/sasrec.py``. Trained with BPR loss over
sampled negatives at each position, evaluated on the shared 1-vs-99 protocol. Writes:

    results/sasrec/test_metrics_d{dim}_L{K}.json
    results/sasrec/rating_metrics_d{dim}_L{K}.json
    results/sasrec/best_model_d{dim}_L{K}.pt
    logs/sasrec/metrics_d{dim}_L{K}.csv

Usage:
    python -m src.train_sasrec --config configs/sasrec.yaml --kcore 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_split, get_n_users_items
from src.data.sequential import get_sequential_dataloaders
from src.evaluation.rating import (
    mae_from_predictions,
    rmse_from_predictions,
)
from src.models.sasrec import SASRec
from src.utils.io import load_config, save_checkpoint, load_checkpoint
from src.utils.metrics_logger import MetricsLogger
from src.utils.seed import set_seed


# Ranking evaluator (SASRec needs the full sequence, not just (u, i))
def sasrec_rank_eval(
    model,
    loader,
    device: str,
    k_values=(5, 10, 20),
) -> dict[str, float]:
    model.eval()
    metrics = {f"HR@{k}": 0.0 for k in k_values}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_values})
    n = 0
    with torch.no_grad():
        for _users, seq, cands_shifted, _raw in loader:
            seq = seq.to(device)
            cands_shifted = cands_shifted.to(device)
            scores = model.score_candidates(seq, cands_shifted)
            _, order = torch.sort(scores, dim=1, descending=True)
            ranks = (order == 0).float().argmax(dim=1).cpu().numpy()
            n += len(ranks)
            for k in k_values:
                hit = (ranks < k).astype(np.float32)
                metrics[f"HR@{k}"] += float(hit.sum())
                metrics[f"NDCG@{k}"] += float(np.where(
                    ranks < k, 1.0 / np.log2(ranks + 2), 0.0
                ).sum())
    return {k: v / max(n, 1) for k, v in metrics.items()}


# Score to rating calibration for SASRec (same linear fit as LightGCN path)
def calibrate_sasrec(model, sequences, max_seqlen, device, val_df, test_df):
    """Score arbitrary (u, i) pairs via SASRec's last-position + candidate
    embedding, fit linear rating = a*score + b on val, evaluate on test.

    Robust to SASRec's near-constant score distributions (falls back to
    train-mean predictor if lstsq fails or scores have near-zero variance).
    """
    def score_fn(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        out = np.empty(len(users), dtype=np.float32)
        BS = 1024
        model.eval()
        with torch.no_grad():
            for start in range(0, len(users), BS):
                end = min(start + BS, len(users))
                us = users[start:end]; iis = items[start:end]
                seqs = []
                for u in us:
                    hist = sequences.get(int(u), [])
                    hs = [i + 1 for i in hist[-max_seqlen:]]
                    seqs.append([0] * (max_seqlen - len(hs)) + hs)
                seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
                cand_t = torch.tensor(iis + 1, dtype=torch.long,
                                      device=device).unsqueeze(1)
                scores = model.score_candidates(seq_t, cand_t).squeeze(1)
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
        "mae_calibrated": mae_from_predictions(pred, ty),
        "calibration_a": float(a),
        "calibration_b": float(b),
        "n": int(len(ty)),
        "val_score_std": vs_std,
    }
    if note:
        out["fallback_note"] = note
    return out



# Trainer
def pick_device() -> torch.device:
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            dev = torch.device("cuda")
        except RuntimeError:
            pass
    return dev


def bpr_step(model, seq, pos, neg) -> tuple[torch.Tensor, float]:
    pos_s, neg_s = model(seq, pos, neg)
    loss = -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()
    return loss, loss.item()


def train(config: dict, kcore_dir: str):
    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    sched_cfg = config.get("scheduler", {})
    eval_cfg = config.get("evaluation", {})
    neg_cfg = config.get("negative_sampling", {})
    paths = config.get("paths", {})

    device = pick_device()

    embed_dim  = int(model_cfg.get("embedding_dim", 128))
    max_seqlen = int(model_cfg.get("max_seqlen", 100))
    num_heads  = int(model_cfg.get("num_heads", 2))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout    = float(model_cfg.get("dropout", 0.2))

    # Loaders.
    loaders = get_sequential_dataloaders(
        kcore_dir,
        batch_size=int(train_cfg.get("batch_size", 256)),
        max_seqlen=max_seqlen,
        num_negatives=int(neg_cfg.get("num_negatives", 1)),
        eval_negatives=int(eval_cfg.get("num_negatives", 99)),
        seed=42,
    )
    n_users = loaders["n_users"]; n_items = loaders["n_items"]
    print(f"Dataset: {n_users:,} users, {n_items:,} items, "
          f"train batches={len(loaders['train'])}")

    model = SASRec(
        n_items=n_items, embed_dim=embed_dim, max_seqlen=max_seqlen,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | dim={embed_dim} | seqlen={max_seqlen} "
          f"| layers={num_layers} | heads={num_heads} | params={n_params:,}")

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 30))
    scheduler = None
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-5)),
        )

    k_values = eval_cfg.get("top_k", [5, 10, 20])
    track_key = f"HR@{k_values[1] if len(k_values) > 1 else k_values[0]}"
    patience = int(train_cfg.get("patience", epochs))

    log_dir = paths.get("log_dir", "logs/sasrec")
    ckpt_dir = paths.get("checkpoint_dir", "results/sasrec")
    run_suffix = f"d{embed_dim}_L{num_layers}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(log_dir, filename=f"metrics_{run_suffix}.csv")
    best_path = os.path.join(ckpt_dir, f"best_model_{run_suffix}.pt")

    best_metric = 0.0; best_epoch = 0; patience_c = 0
    print(f"\nTraining {epochs} epochs (lr={lr}, bs={train_cfg.get('batch_size',256)})...")
    print("=" * 72)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        ep_t = time.time()
        for seq, pos, neg in loaders["train"]:
            seq = seq.to(device); pos = pos.to(device); neg = neg.to(device)
            if neg.dim() == 2 and neg.shape[1] == 1:
                neg = neg.squeeze(1)
            loss, loss_v = bpr_step(model, seq, pos, neg)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss_v; n += 1
        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        val_m = sasrec_rank_eval(model, loaders["val"], str(device), tuple(k_values))
        val_m["train_loss"] = total_loss / max(n, 1)
        val_m["lr"] = cur_lr
        val_m["epoch_time_s"] = round(time.time() - ep_t, 2)
        logger.log(epoch, val_m)

        cur = val_m[track_key]
        marker = ""
        if cur > best_metric:
            best_metric = cur; best_epoch = epoch; patience_c = 0
            save_checkpoint(model, optimizer, epoch, best_path)
            marker = " *"
        else:
            patience_c += 1

        show = ("HR@5", "HR@10", "HR@20", "NDCG@10", "train_loss")
        metric_str = "  ".join(f"{k}: {val_m[k]:.4f}" for k in show if k in val_m)
        print(f"  [{epoch:3d}/{epochs}]  {metric_str}  "
              f"lr={cur_lr:.1e}  t={val_m['epoch_time_s']}s{marker}")

        if patience_c >= patience:
            print(f"Early stop at epoch {epoch} (no gain for {patience} epochs)")
            break

    # Reload best, eval on test, calibrate.
    if os.path.exists(best_path):
        model, _ = load_checkpoint(best_path, model)
        model = model.to(device)
    model.eval()
    test_m = sasrec_rank_eval(model, loaders["test"], str(device), tuple(k_values))

    test_m[f"best_val_{track_key}"] = best_metric
    test_m["best_epoch"] = best_epoch
    test_m["embed_dim"] = embed_dim
    test_m["num_layers"] = num_layers
    test_m["num_heads"] = num_heads
    test_m["max_seqlen"] = max_seqlen
    test_m["total_train_time_s"] = round(time.time() - t0, 2)

    print("\n" + "=" * 72)
    print(f"SASRec test results:")
    for k in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
        if k in test_m:
            print(f"  {k}: {test_m[k]:.4f}")
    print(f"  best val {track_key}: {best_metric:.4f} @ epoch {best_epoch}")
    print(f"  total time: {test_m['total_train_time_s']}s")

    with open(os.path.join(ckpt_dir, f"test_metrics_{run_suffix}.json"), "w") as f:
        json.dump(test_m, f, indent=2)

    # Calibrated RMSE.
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")
    rating_m = calibrate_sasrec(
        model, loaders["sequences"], max_seqlen, str(device), val_df, test_df,
    )
    rating_m["embed_dim"] = embed_dim
    rating_m["num_layers"] = num_layers
    with open(os.path.join(ckpt_dir, f"rating_metrics_{run_suffix}.json"), "w") as f:
        json.dump(rating_m, f, indent=2)
    print(f"SASRec calibrated: RMSE={rating_m['rmse_calibrated']:.4f}  "
          f"MAE={rating_m['mae_calibrated']:.4f}  "
          f"(a={rating_m['calibration_a']:.4f}, b={rating_m['calibration_b']:.4f})")

    print(f"\nSaved:")
    print(f"  {best_path}")
    print(f"  {os.path.join(ckpt_dir, f'test_metrics_{run_suffix}.json')}")
    print(f"  {os.path.join(ckpt_dir, f'rating_metrics_{run_suffix}.json')}")


def main():
    parser = argparse.ArgumentParser(description="Train SASRec on HotelRec")
    parser.add_argument("--config", default="configs/sasrec.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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
        raise FileNotFoundError(kcore_dir)

    train(config, kcore_dir)


if __name__ == "__main__":
    main()
