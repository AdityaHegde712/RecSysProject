"""
Compute RMSE / MAE on the held-out test set.

Two paths:
  1. Native rating prediction  (ItemKNN, Popularity):
       ItemKNN    : weighted average of training ratings over top-k similar items.
       Popularity : predicted rating = mean training rating of the item
                    (falls back to global mean if item unseen in train).
  2. Score calibration         (GMF, LightGCN, TextNCF — ranking-only):
       fits a linear map  rating = a*score + b  on the validation split,
       then reports rmse_calibrated / mae_calibrated on test.

Writes:
  results/baselines/rating_metrics_20core.json
  results/lightgcn/rating_metrics_L{K}.json  (if a LightGCN checkpoint exists)

Usage:
  python scripts/compute_rmse.py --kcore 20
  python scripts/compute_rmse.py --kcore 20 --lightgcn-layers 1   # also do LightGCN
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Allow running as a plain script: `python scripts/compute_rmse.py`
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.data.dataset import load_split, get_n_users_items
from src.evaluation.rating import (
    rmse_from_predictions,
    mae_from_predictions,
    evaluate_rating_calibrated,
)
from src.models.knn import ItemKNN


# -----------------------------------------------------------------------------
# Rating predictors
# -----------------------------------------------------------------------------

def itemknn_predict_ratings_batch(knn: ItemKNN, users: np.ndarray, items: np.ndarray,
                                  global_mean: float) -> np.ndarray:
    """Weighted-average rating prediction for ItemKNN over training neighbors.

    r_hat(u,i) = sum_{j in N(i) intersect H(u)} sim(i,j) * r(u,j)
               / sum_{j in N(i) intersect H(u)} sim(i,j)

    Falls back to global mean when the intersection is empty.
    """
    sim = knn.sim          # (n_items, n_items) sparse, row i = neighbors of i
    user_item = knn.user_item  # (n_users, n_items) sparse, user's rated items/ratings
    binary_ui = (user_item > 0).astype(np.float32)

    preds = np.empty(len(users), dtype=np.float32)
    for k, (u, i) in enumerate(zip(users, items)):
        u = int(u); i = int(i)
        if u >= knn.n_users or i >= knn.n_items:
            preds[k] = global_mean
            continue
        # neighbor similarities for item i (1, n_items)
        sim_i = sim.getrow(i)
        # user's ratings (1, n_items)
        ui_u = user_item.getrow(u)
        bi_u = binary_ui.getrow(u)
        # Elementwise product: sim * rating, summed across j
        num = sim_i.multiply(ui_u).sum()
        den = sim_i.multiply(bi_u).sum()
        if den > 0:
            preds[k] = num / den
        else:
            preds[k] = global_mean
    return np.clip(preds, 1.0, 5.0)


def fit_popularity_rating(train_df: pd.DataFrame, n_items: int) -> tuple[np.ndarray, float]:
    """Return (item_mean_rating[n_items], global_mean)."""
    global_mean = float(train_df["rating"].mean())
    item_sum = train_df.groupby("item_id")["rating"].sum()
    item_cnt = train_df.groupby("item_id")["rating"].count()
    item_mean = (item_sum / item_cnt).to_dict()
    arr = np.full(n_items, global_mean, dtype=np.float32)
    for i, v in item_mean.items():
        if 0 <= int(i) < n_items:
            arr[int(i)] = float(v)
    return arr, global_mean


def popularity_predict_ratings(item_mean: np.ndarray, items: np.ndarray,
                               global_mean: float) -> np.ndarray:
    preds = item_mean[items].astype(np.float32)
    preds = np.where(np.isfinite(preds), preds, global_mean)
    return np.clip(preds, 1.0, 5.0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--lightgcn-layers", type=int, default=None,
                        help="Also compute calibrated RMSE for LightGCN with this K "
                             "(checkpoint must exist at results/lightgcn/best_model_L{K}.pt)")
    parser.add_argument("--lightgcn-dim", type=int, default=64)
    parser.add_argument("--lightgcn-ckpt", default=None,
                        help="Override LightGCN checkpoint path")
    parser.add_argument("--knn-k", type=int, default=20,
                        help="k_neighbors for ItemKNN rating predictor (default 20, "
                             "matches the shipped baselines JSON)")
    args = parser.parse_args()

    kcore_dir = os.path.join(args.data_dir, f"{args.kcore}core")
    train_df = load_split(kcore_dir, "train")
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")
    n_users, n_items = get_n_users_items(kcore_dir)

    print(f"Splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"Users: {n_users:,}  Items: {n_items:,}")
    print(f"Test rating range: [{test_df['rating'].min():.2f}, {test_df['rating'].max():.2f}]")
    print()

    truths = test_df["rating"].astype(np.float32).values
    test_users = test_df["user_id"].astype(np.int64).values
    test_items = test_df["item_id"].astype(np.int64).values

    # ---- Global-mean sanity baseline ----
    global_mean = float(train_df["rating"].mean())
    gm_preds = np.full_like(truths, global_mean, dtype=np.float32)
    gm = {
        "rmse": rmse_from_predictions(gm_preds, truths),
        "mae": mae_from_predictions(gm_preds, truths),
        "n": int(len(truths)),
        "global_mean": global_mean,
    }
    print(f"GlobalMean  : RMSE={gm['rmse']:.4f}  MAE={gm['mae']:.4f}")

    # ---- Popularity (item-mean-rating) ----
    item_mean, _ = fit_popularity_rating(train_df, n_items)
    pop_preds = popularity_predict_ratings(item_mean, test_items, global_mean)
    pop = {
        "rmse": rmse_from_predictions(pop_preds, truths),
        "mae": mae_from_predictions(pop_preds, truths),
        "n": int(len(truths)),
    }
    print(f"Popularity  : RMSE={pop['rmse']:.4f}  MAE={pop['mae']:.4f}")

    # ---- ItemKNN (weighted neighbor rating) ----
    # NOTE: the 20-core split keeps multiple reviews per (user, item) when
    # users updated their TripAdvisor reviews. The stock ItemKNN.fit() sums
    # duplicate ratings via scipy CSR, which is harmless for ranking (relative
    # order is preserved) but produces inflated rating predictions. For RMSE
    # only, we dedupe by taking the mean rating per (user, item) pair.
    k_neighbors = args.knn_k
    print(f"Fitting ItemKNN(k={k_neighbors}) on deduplicated train (mean per u,i)...")
    train_dedup = (train_df.groupby(["user_id", "item_id"], as_index=False)
                          ["rating"].mean())
    print(f"  dedup: {len(train_df):,} -> {len(train_dedup):,} unique (u,i) pairs")
    knn = ItemKNN(k=k_neighbors, n_users=n_users, n_items=n_items)
    knn.fit(train_dedup, n_users=n_users, n_items=n_items)
    print("Scoring test ratings...")
    knn_preds = itemknn_predict_ratings_batch(knn, test_users, test_items, global_mean)
    knn_m = {
        "rmse": rmse_from_predictions(knn_preds, truths),
        "mae": mae_from_predictions(knn_preds, truths),
        "n": int(len(truths)),
        "k_neighbors": k_neighbors,
    }
    print(f"ItemKNN     : RMSE={knn_m['rmse']:.4f}  MAE={knn_m['mae']:.4f}")

    out = {
        "GlobalMean": gm,
        "Popularity": pop,
        "ItemKNN": knn_m,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"rating_metrics_{args.kcore}core.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    # ---- LightGCN (calibrated score -> rating) ----
    if args.lightgcn_layers is not None:
        print()
        print(f"Computing calibrated RMSE for LightGCN (K={args.lightgcn_layers})...")
        from src.models.lightgcn import LightGCN, build_norm_adj
        from src.utils.io import load_checkpoint

        adj = build_norm_adj(
            train_df["user_id"].values.astype(np.int64),
            train_df["item_id"].values.astype(np.int64),
            n_users, n_items,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            torch.zeros(1, device=device)
        except Exception:
            device = "cpu"
        adj = adj.to(device)

        model = LightGCN(
            num_users=n_users, num_items=n_items,
            embed_dim=args.lightgcn_dim, num_layers=args.lightgcn_layers,
            adj_hat=adj,
        ).to(device)
        ckpt = args.lightgcn_ckpt or f"results/lightgcn/best_model_L{args.lightgcn_layers}.pt"
        print(f"Loading checkpoint: {ckpt}")
        model, _ = load_checkpoint(ckpt, model)
        model = model.to(device)
        model.eval()
        model.cache_embeddings()

        cal = evaluate_rating_calibrated(model, val_df, test_df, device=device)
        cal["num_layers"] = args.lightgcn_layers
        cal["embed_dim"] = args.lightgcn_dim
        out_lg_dir = "results/lightgcn"
        Path(out_lg_dir).mkdir(parents=True, exist_ok=True)
        out_lg = os.path.join(out_lg_dir, f"rating_metrics_L{args.lightgcn_layers}.json")
        with open(out_lg, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"LightGCN(K={args.lightgcn_layers}) calibrated: "
              f"RMSE={cal['rmse_calibrated']:.4f}  MAE={cal['mae_calibrated']:.4f}  "
              f"(a={cal['calibration_a']:.4f}, b={cal['calibration_b']:.4f})")
        print(f"Saved: {out_lg}")


if __name__ == "__main__":
    main()
