"""
Compute RMSE / MAE on the held-out test set.

Two paths:
  1. Native rating prediction  (ItemKNN, Popularity):
       ItemKNN    : weighted average of training ratings over top-k similar items.
       Popularity : predicted rating = mean training rating of the item (falls back to global mean if item unseen in train).
  2. Score calibration         (GMF, LightGCN-HG, SASRec -- ranking-only): fits a linear map  rating = a*score + b  on the 
        validation split, then reports rmse_calibrated / mae_calibrated on test.

Writes:
  results/baselines/rating_metrics_20core.json
  results/gmf/rating_metrics.json  (if --gmf-ckpt given)
  results/lightgcn_hg/rating_metrics_L{L}_d{D}_{tiers}.json  (if --lightgcn-hg-ckpt given)
  results/text_ncf/rating_metrics.json              (if --text-ncf-ckpt given)
  results/text_ncf_mt/rating_metrics.json           (if --text-ncf-mt-ckpt given)
  results/text_ncf_subrating/rating_metrics.json    (if --text-ncf-subrating-ckpt given)

SASRec's calibrated RMSE is produced directly by src/train_sasrec.py alongside its training run, so it isn't re-computed here.

Usage:
  python scripts/compute_rmse.py --kcore 20
  python scripts/compute_rmse.py --kcore 20 \
      --gmf-ckpt results/gmf/best_model.pt --gmf-dim 64
  python scripts/compute_rmse.py --kcore 20 \
      --lightgcn-hg-ckpt results/lightgcn_hg/best_model_L1_d256_grc.pt \
      --lightgcn-hg-dim 256 --lightgcn-hg-layers 1
  python scripts/compute_rmse.py --kcore 20 \
      --text-ncf-ckpt results/text_ncf/best_model.pt \
      --text-ncf-config configs/text_ncf.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_split, get_n_users_items
from src.evaluation.rating import (
    rmse_from_predictions,
    mae_from_predictions,
    evaluate_rating_calibrated,
)
from src.models.knn import ItemKNN
from src.utils.io import load_config


# Rating predictors
def itemknn_predict_ratings_batch(knn: ItemKNN, users: np.ndarray, items: np.ndarray, global_mean: float) -> np.ndarray:
    """Weighted-average rating prediction for ItemKNN over training neighbors.

    r_hat(u,i) = sum_{j in N(i) intersect H(u)} sim(i,j) * r(u,j)
               / sum_{j in N(i) intersect H(u)} sim(i,j)

    Falls back to global mean when the intersection is empty.
    """
    sim = knn.sim
    user_item = knn.user_item
    binary_ui = (user_item > 0).astype(np.float32)

    preds = np.empty(len(users), dtype=np.float32)
    for k, (u, i) in enumerate(zip(users, items)):
        u = int(u); i = int(i)
        if u >= knn.n_users or i >= knn.n_items:
            preds[k] = global_mean
            continue
        sim_i = sim.getrow(i)
        ui_u = user_item.getrow(u)
        bi_u = binary_ui.getrow(u)
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


def popularity_predict_ratings(item_mean: np.ndarray, items: np.ndarray, global_mean: float) -> np.ndarray:
    preds = item_mean[items].astype(np.float32)
    preds = np.where(np.isfinite(preds), preds, global_mean)
    return np.clip(preds, 1.0, 5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results/baselines")
    parser.add_argument("--knn-k", type=int, default=20,
                        help="k_neighbors for ItemKNN rating predictor (default 20, "
                             "matches the shipped baselines JSON)")
    parser.add_argument("--gmf-ckpt", default=None,
                        help="Path to a GMF checkpoint; if given, compute GMF's "
                             "calibrated RMSE. Default None skips GMF.")
    parser.add_argument("--gmf-dim", type=int, default=64,
                        help="GMF embedding dim (must match the checkpoint)")
    parser.add_argument("--lightgcn-hg-ckpt", default=None,
                        help="Path to a LightGCN-HG checkpoint; if given, "
                             "compute calibrated RMSE for the HG variant.")
    parser.add_argument("--lightgcn-hg-dim", type=int, default=256,
                        help="LightGCN-HG embedding dim (must match checkpoint)")
    parser.add_argument("--lightgcn-hg-layers", type=int, default=1)
    parser.add_argument("--lightgcn-hg-tiers", type=str,
                        default="g_id,region_slug,country_slug",
                        help="Tier list used when the HG checkpoint was trained; "
                             "must match the graph used at training time.")
    parser.add_argument("--hotel-meta",
                        default="data/processed/hotel_meta/hotel_meta.parquet",
                        help="Path to hotel metadata parquet from "
                             "scripts/extract_hotel_meta.py")
    parser.add_argument("--lightgcn-hg-out", default=None,
                        help="Override output path for HG rating metrics.")
    parser.add_argument("--text-ncf-ckpt", default=None,
                        help="Path to a TextNCF checkpoint; computes calibrated RMSE.")
    parser.add_argument("--text-ncf-config", default="configs/text_ncf.yaml")
    parser.add_argument("--text-ncf-mt-ckpt", default=None,
                        help="Path to a Multi-Task TextNCF checkpoint; "
                             "computes calibrated RMSE.")
    parser.add_argument("--text-ncf-mt-config", default="configs/text_ncf_mt.yaml")
    parser.add_argument("--text-ncf-subrating-ckpt", default=None,
                        help="Path to a Sub-Rating TextNCF checkpoint; "
                             "computes calibrated RMSE.")
    parser.add_argument("--text-ncf-subrating-config",
                        default="configs/text_ncf_subrating.yaml")
    args = parser.parse_args()

    kcore_dir = os.path.join(args.data_dir, f"{args.kcore}core")
    train_df = load_split(kcore_dir, "train")
    val_df = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")
    n_users, n_items = get_n_users_items(kcore_dir)

    print(f"Splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"Users: {n_users:,}  Items: {n_items:,}")
    print()

    truths = test_df["rating"].astype(np.float32).values
    test_users = test_df["user_id"].astype(np.int64).values
    test_items = test_df["item_id"].astype(np.int64).values

    # Global-mean sanity baseline
    global_mean = float(train_df["rating"].mean())
    gm_preds = np.full_like(truths, global_mean, dtype=np.float32)
    gm = {
        "rmse": rmse_from_predictions(gm_preds, truths),
        "mae": mae_from_predictions(gm_preds, truths),
        "n": int(len(truths)),
        "global_mean": global_mean,
    }
    print(f"GlobalMean  : RMSE={gm['rmse']:.4f}  MAE={gm['mae']:.4f}")

    # Popularity (item-mean-rating)
    item_mean, _ = fit_popularity_rating(train_df, n_items)
    pop_preds = popularity_predict_ratings(item_mean, test_items, global_mean)
    pop = {
        "rmse": rmse_from_predictions(pop_preds, truths),
        "mae": mae_from_predictions(pop_preds, truths),
        "n": int(len(truths)),
    }
    print(f"Popularity  : RMSE={pop['rmse']:.4f}  MAE={pop['mae']:.4f}")

    # ItemKNN (weighted neighbor rating)
    # NOTE: the 20-core split keeps multiple reviews per (user, item) when users updated their TripAdvisor reviews. The stock ItemKNN.fit() sums
    # duplicate ratings via scipy CSR, which is harmless for ranking (relative order is preserved) but produces inflated rating predictions. For RMSE
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

    out = {"GlobalMean": gm, "Popularity": pop, "ItemKNN": knn_m}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"rating_metrics_{args.kcore}core.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    # GMF (calibrated score -> rating)
    if args.gmf_ckpt is not None:
        print()
        print("Computing calibrated RMSE for GMF...")
        from src.models.gmf import GMF
        from src.utils.io import load_checkpoint

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            torch.zeros(1, device=device)
        except Exception:
            device = "cpu"

        gmf = GMF(num_users=n_users, num_items=n_items, embed_dim=args.gmf_dim).to(device)
        print(f"Loading checkpoint: {args.gmf_ckpt}")
        gmf, _ = load_checkpoint(args.gmf_ckpt, gmf)
        gmf = gmf.to(device)
        gmf.eval()

        cal = evaluate_rating_calibrated(gmf, val_df, test_df, device=device)
        cal["embed_dim"] = args.gmf_dim
        out_gmf_dir = "results/gmf"
        Path(out_gmf_dir).mkdir(parents=True, exist_ok=True)
        out_gmf = os.path.join(out_gmf_dir, "rating_metrics.json")
        with open(out_gmf, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"GMF calibrated: RMSE={cal['rmse_calibrated']:.4f}  "
              f"MAE={cal['mae_calibrated']:.4f}  "
              f"(a={cal['calibration_a']:.4f}, b={cal['calibration_b']:.4f})")
        print(f"Saved: {out_gmf}")

    # LightGCN-HG (calibrated score -> rating)
    if args.lightgcn_hg_ckpt is not None:
        print()
        print("Computing calibrated RMSE for LightGCN-HG ...")
        from src.models.lightgcn_hg import LightGCNHG, build_hg_norm_adj
        from src.utils.io import load_checkpoint

        raw_tiers = args.lightgcn_hg_tiers.strip()
        tiers = [] if raw_tiers.lower() == "none" else [
            t.strip() for t in raw_tiers.split(",") if t.strip()
        ]

        if tiers:
            if not os.path.exists(args.hotel_meta):
                raise FileNotFoundError(
                    f"Missing {args.hotel_meta} -- run "
                    f"python -m scripts.extract_hotel_meta --kcore {args.kcore}"
                )
            hotel_meta = pd.read_parquet(args.hotel_meta)
        else:
            hotel_meta = pd.DataFrame(columns=["item_id"])

        adj_hat, graph_meta = build_hg_norm_adj(
            train_df["user_id"].values.astype(np.int64),
            train_df["item_id"].values.astype(np.int64),
            hotel_meta, n_users, n_items, tiers=tiers,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            torch.zeros(1, device=device)
        except Exception:
            device = "cpu"
        adj_hat = adj_hat.to(device)

        hg = LightGCNHG(
            num_users=n_users, num_items=n_items,
            graph_meta=graph_meta,
            embed_dim=args.lightgcn_hg_dim,
            num_layers=args.lightgcn_hg_layers,
            adj_hat=adj_hat,
        ).to(device)
        print(f"Loading checkpoint: {args.lightgcn_hg_ckpt}")
        hg, _ = load_checkpoint(args.lightgcn_hg_ckpt, hg)
        hg = hg.to(device)
        hg.eval()
        hg.cache_embeddings()

        cal = evaluate_rating_calibrated(hg, val_df, test_df, device=device)
        cal["num_layers"] = args.lightgcn_hg_layers
        cal["embed_dim"] = args.lightgcn_hg_dim
        cal["tiers"] = tiers

        TIER_SHORT = {"g_id": "g", "region_slug": "r", "country_slug": "c"}
        tsuf = "".join(TIER_SHORT[t] for t in tiers) if tiers else "none"
        out_hg = args.lightgcn_hg_out or os.path.join(
            "results", "lightgcn_hg",
            f"rating_metrics_L{args.lightgcn_hg_layers}_d{args.lightgcn_hg_dim}_{tsuf}.json",
        )
        Path(os.path.dirname(out_hg)).mkdir(parents=True, exist_ok=True)
        with open(out_hg, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"LightGCN-HG calibrated: RMSE={cal['rmse_calibrated']:.4f}  "
              f"MAE={cal['mae_calibrated']:.4f}  "
              f"(a={cal['calibration_a']:.4f}, b={cal['calibration_b']:.4f})")
        print(f"Saved: {out_hg}")

    # TextNCF variants (calibrated score -> rating)
    _run_text_ncf_calibration(args, val_df, test_df, n_users, n_items)


def _pick_torch_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.zeros(1, device=device)
    except Exception:
        device = "cpu"
    return device


def _load_text_ncf_variant(variant_ckpt, variant_config, variant_cls,
                           n_users, n_items, device, extra_kwargs=None):
    """Load a TextNCF-family model (base / MT / Subrating) with text embeddings.
    variant_cls: class object (TextNCF, TextNCFMultiTask, TextNCFSubrating).
    """
    from src.data.text_embeddings import load_text_embeddings, TEXT_EMB_DIR
    from src.utils.io import load_checkpoint

    cfg = load_config(variant_config)
    mcfg = cfg.get("model", {})
    kwargs = dict(
        num_users=n_users,
        num_items=n_items,
        embed_dim=mcfg.get("embed_dim", 64),
        text_dim=mcfg.get("text_dim", 384),
        text_proj_dim=mcfg.get("text_proj_dim", 64),
        mlp_layers=mcfg.get("mlp_layers", [128, 64]),
        dropout=mcfg.get("dropout", 0.2),
        use_gmf=mcfg.get("use_gmf", True),
        use_text=mcfg.get("use_text", True),
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    model = variant_cls(**kwargs)

    emb_dir = cfg.get("paths", {}).get("text_emb_dir", TEXT_EMB_DIR)
    u_emb, i_emb = load_text_embeddings(emb_dir)
    model.set_text_embeddings(torch.from_numpy(u_emb), torch.from_numpy(i_emb))

    load_checkpoint(variant_ckpt, model=model)
    model.to(device)
    model.eval()
    return model, mcfg


def _run_text_ncf_calibration(args, val_df, test_df, n_users, n_items):
    specs = []
    if args.text_ncf_ckpt is not None:
        from src.models.text_ncf import TextNCF
        specs.append(("text_ncf", TextNCF, args.text_ncf_ckpt,
                      args.text_ncf_config, None))
    if args.text_ncf_mt_ckpt is not None:
        from src.models.text_ncf_mt import TextNCFMultiTask
        specs.append(("text_ncf_mt", TextNCFMultiTask,
                      args.text_ncf_mt_ckpt, args.text_ncf_mt_config, None))
    if args.text_ncf_subrating_ckpt is not None:
        from src.models.text_ncf_subrating import TextNCFSubrating
        cfg = load_config(args.text_ncf_subrating_config).get("model", {})
        extra = {
            "num_aspects": cfg.get("num_aspects", 6),
            "aspect_hidden": cfg.get("aspect_hidden", 32),
        }
        specs.append(("text_ncf_subrating", TextNCFSubrating,
                      args.text_ncf_subrating_ckpt,
                      args.text_ncf_subrating_config, extra))

    if not specs:
        return

    device = _pick_torch_device()

    for name, cls, ckpt, cfg_path, extra in specs:
        print()
        print(f"Computing calibrated RMSE for {name}...")
        model, mcfg = _load_text_ncf_variant(
            ckpt, cfg_path, cls, n_users, n_items, device, extra_kwargs=extra)

        cal = evaluate_rating_calibrated(model, val_df, test_df, device=device)
        cal["model"] = name
        cal["embed_dim"] = mcfg.get("embed_dim", 64)
        cal["text_proj_dim"] = mcfg.get("text_proj_dim", 64)

        out_dir = os.path.join("results", name)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, "rating_metrics.json")
        with open(out_path, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"{name} calibrated: RMSE={cal['rmse_calibrated']:.4f}  "
              f"MAE={cal['mae_calibrated']:.4f}  "
              f"(a={cal['calibration_a']:.4f}, b={cal['calibration_b']:.4f})")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
