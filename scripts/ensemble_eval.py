"""
Score ensemble: LightGCN + ItemKNN, linear combination with per-user
min-max normalization.

Rationale: on HotelRec the two models have different operating regions --
LightGCN wins HR@10 / HR@20 (broad top-k recall), ItemKNN wins NDCG@5 at
k_neighbors=20 (concentrated top-1 placement). The goal here is to see
whether a weighted sum of per-user normalized scores beats either endpoint.

Procedure:
  1. Score every (user, candidate) pair with both models on val + test.
  2. Normalize each user's 100-candidate scores to [0,1] via min-max, per model.
  3. Sweep w in [0, 1] on the validation split, pick w* that maximizes HR@10.
  4. Report test metrics at w*, plus the ItemKNN (w=0) and LightGCN (w=1)
     endpoints for context.

Usage:
  python scripts/ensemble_eval.py --kcore 20 \
      --lightgcn-layers 1 --lightgcn-dim 256 \
      --lightgcn-ckpt results/lightgcn/best_model_L1_d256.pt \
      --knn-k 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as a plain script
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import get_dataloaders, get_n_users_items, load_split
from src.evaluation.ranking import hit_ratio, ndcg
from src.models.knn import ItemKNN
from src.models.lightgcn import LightGCN, build_norm_adj
from src.utils.io import load_checkpoint


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def minmax_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Per-row min-max normalization to [0, 1].

    Rows with zero range (all identical scores -- usually an empty-history
    user for ItemKNN) are mapped to all-zeros, which is fine because they
    contribute nothing to the ensemble's discriminative power for that user.
    """
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    rng = hi - lo
    # Avoid div-by-zero: where rng == 0, output zeros.
    safe = np.where(rng > 0, rng, 1.0)
    out = (x - lo) / safe
    out = np.where(rng > 0, out, 0.0)
    return out.astype(np.float32)


def score_with_lightgcn(model, loader, device: str) -> np.ndarray:
    """Return (N, 100) array of LightGCN scores for the 100 candidates per row."""
    model.eval()
    model.cache_embeddings()
    all_scores = []
    with torch.no_grad():
        for users, items, _ in tqdm(loader, desc="lgcn", leave=False):
            bs, nc = items.shape
            u_flat = users.unsqueeze(1).expand(-1, nc).reshape(-1).to(device)
            i_flat = items.reshape(-1).to(device)
            s = model(u_flat, i_flat).reshape(bs, nc)
            all_scores.append(s.cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def score_with_itemknn(knn: ItemKNN, loader) -> np.ndarray:
    """Return (N, 100) array of ItemKNN scores for the 100 candidates per row."""
    all_scores = []
    for users, items, _ in tqdm(loader, desc="knn", leave=False):
        s = knn.predict_batch(users, items).numpy()
        all_scores.append(s)
    return np.concatenate(all_scores, axis=0)


def gather_positions(loader) -> np.ndarray:
    """Return (N,) array -- the ground-truth item is always position 0 in items."""
    all_items_row0 = []
    for _, items, _ in loader:
        all_items_row0.append(items[:, 0].numpy())
    return np.concatenate(all_items_row0, axis=0)


def metrics_from_scores(
    scores: np.ndarray,
    items_matrix_per_batch: list[np.ndarray],
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Given (N, 100) combined scores and the 100 candidate item IDs per row,
    compute HR@k and NDCG@k using the SAME ranking primitive as the shared
    evaluate_ranking() (torch.sort with default, non-stable behavior).

    Why not np.argsort? For users with tied scores (e.g. ItemKNN returning
    all-zeros because they share no history with any candidate), numpy's
    quicksort preserves natural order, which would put the ground-truth
    item (at candidate index 0) at rank 0 automatically -- a ~20pp HR
    inflation vs torch.sort's non-deterministic tie-breaking. Using
    torch.sort here keeps the ensemble numbers directly comparable to the
    committed baselines JSON.
    """
    metrics = {f"{m}@{k}": [] for k in k_values for m in ("HR", "NDCG")}
    items_all = np.concatenate(items_matrix_per_batch, axis=0)
    scores_t = torch.from_numpy(scores)
    _, idx = torch.sort(scores_t, dim=1, descending=True)
    idx_np = idx.numpy()
    ranked_items = np.take_along_axis(items_all, idx_np, axis=1)
    gt = items_all[:, 0]
    N = len(items_all)

    for b in range(N):
        ranked_list = ranked_items[b].tolist()
        gt_item = int(gt[b])
        for k in k_values:
            metrics[f"HR@{k}"].append(hit_ratio(ranked_list, gt_item, k))
            metrics[f"NDCG@{k}"].append(ndcg(ranked_list, gt_item, k))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def gather_items_matrix(loader) -> np.ndarray:
    """Concatenated (N, 100) item IDs from the loader."""
    mats = []
    for _, items, _ in loader:
        mats.append(items.numpy())
    return np.concatenate(mats, axis=0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--lightgcn-layers", type=int, default=1)
    parser.add_argument("--lightgcn-dim", type=int, default=256)
    parser.add_argument("--lightgcn-ckpt",
                        default="results/lightgcn/best_model_L1_d256.pt")
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sweep-points", type=int, default=21,
                        help="Number of w values to test in [0, 1] on val")
    parser.add_argument("--out-dir", default="results/lightgcn")
    args = parser.parse_args()

    kcore_dir = os.path.join(args.data_dir, f"{args.kcore}core")
    n_users, n_items = get_n_users_items(kcore_dir)
    print(f"Dataset: kcore={args.kcore}, {n_users:,} users, {n_items:,} items")

    # ---- Loaders (share the same 1-vs-99 eval datasets) ----
    loaders = get_dataloaders(
        kcore_dir,
        batch_size=args.batch_size,
        num_negatives=1,
        eval_negatives=99,
        seed=42,
    )
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    print(f"Val: {len(val_loader.dataset):,}, Test: {len(test_loader.dataset):,}")

    # ---- LightGCN ----
    print("\nLoading LightGCN...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.zeros(1, device=device)
    except Exception:
        device = "cpu"

    train_df = load_split(kcore_dir, "train")
    adj = build_norm_adj(
        train_df["user_id"].values.astype(np.int64),
        train_df["item_id"].values.astype(np.int64),
        n_users, n_items,
    ).to(device)

    lgcn = LightGCN(
        num_users=n_users, num_items=n_items,
        embed_dim=args.lightgcn_dim, num_layers=args.lightgcn_layers,
        adj_hat=adj,
    ).to(device)
    lgcn, _ = load_checkpoint(args.lightgcn_ckpt, lgcn)
    lgcn = lgcn.to(device)
    print(f"  loaded {args.lightgcn_ckpt}  (K={args.lightgcn_layers}, "
          f"dim={args.lightgcn_dim}, device={device})")

    # ---- ItemKNN ----
    print(f"\nFitting ItemKNN(k={args.knn_k})...")
    knn = ItemKNN(k=args.knn_k, n_users=n_users, n_items=n_items)
    knn.fit(train_df, n_users=n_users, n_items=n_items)

    # ---- Score val + test with both models ----
    print("\nScoring validation split...")
    val_lgcn = score_with_lightgcn(lgcn, val_loader, device)
    val_knn  = score_with_itemknn(knn, val_loader)
    val_items = gather_items_matrix(val_loader)

    print("Scoring test split...")
    test_lgcn = score_with_lightgcn(lgcn, test_loader, device)
    test_knn  = score_with_itemknn(knn, test_loader)
    test_items = gather_items_matrix(test_loader)

    print(f"\nVal scores:  lgcn shape={val_lgcn.shape},  knn shape={val_knn.shape}")
    print(f"Test scores: lgcn shape={test_lgcn.shape}, knn shape={test_knn.shape}")

    # ---- Normalize per-row ----
    val_lgcn_n = minmax_normalize_rows(val_lgcn)
    val_knn_n  = minmax_normalize_rows(val_knn)
    test_lgcn_n = minmax_normalize_rows(test_lgcn)
    test_knn_n  = minmax_normalize_rows(test_knn)

    # ---- Sweep w on val, maximize HR@10 ----
    print(f"\nSweeping w in [0,1] with {args.sweep_points} points...")
    ws = np.linspace(0.0, 1.0, args.sweep_points)
    val_items_list = [val_items]

    best = {"w": None, "val_hr10": -1.0, "val_metrics": None}
    sweep_log = []
    for w in ws:
        combined = w * val_lgcn_n + (1.0 - w) * val_knn_n
        m = metrics_from_scores(combined, val_items_list)
        sweep_log.append({"w": float(w), **m})
        hr10 = m["HR@10"]
        marker = ""
        if hr10 > best["val_hr10"]:
            best.update(w=float(w), val_hr10=float(hr10), val_metrics=m)
            marker = " *"
        print(f"  w={w:.2f}  HR@5={m['HR@5']:.4f}  HR@10={m['HR@10']:.4f}  "
              f"HR@20={m['HR@20']:.4f}  NDCG@5={m['NDCG@5']:.4f}  "
              f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}{marker}")

    best_w = best["w"]
    print(f"\nBest w on val: {best_w:.3f}  (val HR@10 = {best['val_hr10']:.4f})")

    # ---- Evaluate on test at best_w and at endpoints for reference ----
    test_items_list = [test_items]
    results = {}

    for w_label, w_val in [
        ("itemknn_only (w=0)", 0.0),
        ("lightgcn_only (w=1)", 1.0),
        (f"ensemble (w={best_w:.3f})", best_w),
    ]:
        combined = w_val * test_lgcn_n + (1.0 - w_val) * test_knn_n
        m = metrics_from_scores(combined, test_items_list)
        results[w_label] = m
        print(f"\nTest @ {w_label}:")
        for k in ("HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"):
            print(f"  {k}: {m[k]:.4f}")

    # ---- Save ----
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, "ensemble_test_metrics.json")
    payload = {
        "ensemble_best_w": best_w,
        "val_HR@10_at_best_w": best["val_hr10"],
        "test_metrics": results,
        "val_sweep": sweep_log,
        "config": {
            "lightgcn_layers": args.lightgcn_layers,
            "lightgcn_dim": args.lightgcn_dim,
            "lightgcn_ckpt": args.lightgcn_ckpt,
            "knn_k": args.knn_k,
            "kcore": args.kcore,
            "normalization": "per-row min-max",
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
