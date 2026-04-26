"""Phase 3: LightGBM meta-learner over the four best Phase-2 variants.

Stacks SASRec (Hriday primary), LightGCN-HG (Hriday secondary),
NeuMF-Attn (Aditya enhanced), and TextNCF Multi-Task (Pramod enhanced)
into a single LGBMRanker. Base models are frozen at their best-val
checkpoints; the meta-learner trains on val (1-vs-99) and evaluates on
test using the same protocol every other variant uses.

Usage:
    python -m src.phase3_meta_ensemble --kcore 20

Outputs (under results/phase3_meta/):
    test_metrics.json       - HR@k, NDCG@k for k=5,10,20
    rating_metrics.json     - calibrated RMSE/MAE
    feature_importances.json
    lightgbm.txt            - saved LGBMRanker model
    summary.md              - written by Hriday separately
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure project root is importable when running as `python -m src.phase3_meta_ensemble`
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.dataset import (
    EvalInteractionDataset,
    get_n_users_items,
    get_user_positive_items,
    load_split,
)
from src.data.text_embeddings import TEXT_EMB_DIR, load_text_embeddings
from src.data.sequential import build_user_sequences
from src.evaluation.rating import (
    mae_from_predictions,
    rmse_from_predictions,
)
from src.models.lightgcn_hg import LightGCNHG, build_hg_norm_adj
from src.models.neumf_attn import NeuMF_Attn
from src.models.sasrec import SASRec
from src.models.text_ncf_mt import TextNCFMultiTask
from src.utils.io import load_checkpoint, load_config
from src.utils.seed import set_seed
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Device + paths
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-model loaders. Each returns a `score_pairs(users, items)` callable
# that takes 1D long tensors of equal length and returns a 1D float tensor.
# ---------------------------------------------------------------------------

def _load_sasrec(kcore_dir: str, n_items: int, device: torch.device, seed: int,
                 group_size: int = 100):
    """Returns a score_pairs callable bound to a fixed candidate-group size.

    SASRec is sequence-conditioned, so we have to reshape the flat (user, item)
    score request into per-user blocks of `group_size` candidates and call
    `model.score_candidates(seq, cands)` once per block. `group_size` must
    match the loader's `1 + eval_negatives` (e.g. 100 for 1-vs-99).
    """
    cfg = load_config("configs/sasrec.yaml")
    mcfg = cfg.get("model", {})
    max_seqlen = mcfg.get("max_seqlen", 100)

    model = SASRec(
        n_items=n_items,
        embed_dim=mcfg.get("embedding_dim", mcfg.get("embed_dim", 128)),
        max_seqlen=max_seqlen,
        num_layers=mcfg.get("num_layers", 2),
        num_heads=mcfg.get("num_heads", 2),
        dropout=mcfg.get("dropout", 0.2),
    ).to(device)
    load_checkpoint("results/sasrec/best_model_d128_L2.pt", model=model)
    model.eval()

    # Pre-build per-user sequences ONCE (training history, +1-shifted, padded)
    seqs, _, _ = build_user_sequences(kcore_dir, max_seqlen)

    def get_seq_tensor(user_ids: np.ndarray) -> torch.Tensor:
        rows = []
        for u in user_ids:
            hist = seqs.get(int(u), [])
            shifted = [i + 1 for i in hist[-max_seqlen:]]
            padded = [0] * (max_seqlen - len(shifted)) + shifted
            rows.append(padded)
        return torch.tensor(rows, dtype=torch.long, device=device)

    @torch.no_grad()
    def score_pairs(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # Reshape to per-user blocks of `group_size` candidates so we can use
        # SASRec's batched score_candidates(seq, cands).
        # users: (B,) where B = G * group_size, ordered as (group, candidate).
        # items: (B,) raw item ids
        n = users.numel()
        if n % group_size != 0:
            raise ValueError(
                f"SASRec score_pairs expected a multiple of {group_size} pairs, got {n}. "
                "Pass --eval-negatives matching the loader you constructed _load_sasrec for."
            )
        G = n // group_size
        users_grouped = users.view(G, group_size)
        items_grouped = items.view(G, group_size)
        # All rows in a group share the same user, take col 0
        u_block = users_grouped[:, 0].cpu().numpy()
        seq = get_seq_tensor(u_block)                          # (G, L)
        cands_shifted = items_grouped + 1                       # (G, group_size)
        scores = model.score_candidates(seq, cands_shifted)    # (G, group_size)
        return scores.reshape(-1)

    return score_pairs


def _load_lightgcn_hg(kcore_dir: str, n_users: int, n_items: int,
                      device: torch.device):
    cfg = load_config("configs/lightgcn_hg.yaml")
    mcfg = cfg.get("model", {})
    tiers = list(cfg.get("graph", {}).get("tiers", []))

    train_df = load_split(kcore_dir, "train")
    if tiers:
        meta_path = "data/processed/hotel_meta/hotel_meta.parquet"
        hotel_meta = pd.read_parquet(meta_path)
    else:
        hotel_meta = pd.DataFrame(columns=["item_id"])

    adj_hat, graph_meta = build_hg_norm_adj(
        train_df["user_id"].values.astype(np.int64),
        train_df["item_id"].values.astype(np.int64),
        hotel_meta, n_users, n_items, tiers=tiers,
    )
    adj_hat = adj_hat.to(device)

    model = LightGCNHG(
        num_users=n_users, num_items=n_items, graph_meta=graph_meta,
        embed_dim=mcfg.get("embedding_dim", 256),
        num_layers=mcfg.get("num_layers", 1),
        adj_hat=adj_hat,
    ).to(device)

    # Load grc checkpoint (the headline LightGCN-HG run).
    load_checkpoint("results/lightgcn_hg/best_model_L1_d256_grc.pt", model=model)
    model.eval()
    model.cache_embeddings()

    @torch.no_grad()
    def score_pairs(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # forward(users, items) returns dot product of (cached) user/item
        # embeddings. cache_embeddings() above set _cached_u/_cached_i.
        return model(users, items)

    return score_pairs


def _load_neumf(kcore_dir: str, n_users: int, n_items: int,
                device: torch.device):
    cfg = load_config("configs/neumf_attn.yaml")
    mcfg = cfg.get("model", {})

    # Need item-aspect matrix from train split (exact same logic as
    # train_neumf_attn.build_item_aspects).
    from src.train_neumf_attn import build_item_aspects, SUB_RATING_COLS  # noqa: F401
    train_df = load_split(kcore_dir, "train")
    item_aspects = build_item_aspects(train_df, n_items).to(device)

    model = NeuMF_Attn(
        n_users=n_users, n_items=n_items,
        gmf_dim=mcfg.get("gmf_dim", 64),
        mlp_dim=mcfg.get("mlp_dim", 64),
        mlp_layers=mcfg.get("mlp_layers", [256, 128, 64]),
        dropout=mcfg.get("dropout", 0.2),
        item_aspects=item_aspects,
        use_attention=mcfg.get("use_attention", True),
    ).to(device)
    load_checkpoint("results/neumf_attn/best_model_gmf64_mlp64.pt", model=model)
    model.eval()

    @torch.no_grad()
    def score_pairs(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return model._score(users, items)

    return score_pairs


def _load_text_ncf_mt(n_users: int, n_items: int, device: torch.device):
    cfg = load_config("configs/text_ncf_mt.yaml")
    mcfg = cfg.get("model", {})

    model = TextNCFMultiTask(
        num_users=n_users, num_items=n_items,
        embed_dim=mcfg.get("embed_dim", 64),
        text_dim=mcfg.get("text_dim", 384),
        text_proj_dim=mcfg.get("text_proj_dim", 64),
        mlp_layers=mcfg.get("mlp_layers", [128, 64]),
        dropout=mcfg.get("dropout", 0.2),
        use_gmf=mcfg.get("use_gmf", True),
        use_text=mcfg.get("use_text", True),
    )
    emb_dir = cfg.get("paths", {}).get("text_emb_dir", TEXT_EMB_DIR)
    u_emb, i_emb = load_text_embeddings(emb_dir)
    model.set_text_embeddings(torch.from_numpy(u_emb), torch.from_numpy(i_emb))

    load_checkpoint("results/text_ncf_mt/best_model.pt", model=model)
    model.to(device).eval()

    @torch.no_grad()
    def score_pairs(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return model(users, items)

    return score_pairs


# ---------------------------------------------------------------------------
# Build (val, test) score tables
# ---------------------------------------------------------------------------

MODEL_NAMES = ["sasrec", "lightgcn_hg", "neumf_attn", "text_ncf_mt"]


def build_score_table(loader: DataLoader, scorers: dict, device: torch.device,
                      desc: str = "score"):
    """Iterate the loader once, collect (user, item, label, score_*) per pair.

    Returns a pandas DataFrame with one row per (user, candidate) pair.
    """
    n_models = len(scorers)
    rows_user, rows_item, rows_label = [], [], []
    rows_scores = {name: [] for name in MODEL_NAMES}

    from tqdm import tqdm
    for users, items, labels in tqdm(loader, desc=desc, leave=False):
        # users: (B,)         items: (B, C)      labels: (B, C) one-hot at col 0
        B, C = items.shape
        u_flat = users.unsqueeze(1).expand(-1, C).reshape(-1).to(device)
        i_flat = items.reshape(-1).to(device)
        for name in MODEL_NAMES:
            s = scorers[name](u_flat, i_flat).detach().cpu().numpy()
            rows_scores[name].append(s)

        rows_user.append(u_flat.cpu().numpy())
        rows_item.append(i_flat.cpu().numpy())
        rows_label.append(labels.reshape(-1).numpy())

    df = pd.DataFrame({
        "user_id": np.concatenate(rows_user),
        "item_id": np.concatenate(rows_item),
        "label":   np.concatenate(rows_label).astype(np.int8),
        **{name: np.concatenate(rows_scores[name]).astype(np.float32)
           for name in MODEL_NAMES},
    })
    return df


def per_user_minmax(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Min-max scale each model's column within each user's group of 100."""
    out = df.copy()
    grp = out.groupby("user_id", sort=False)
    for col in score_cols:
        gmin = grp[col].transform("min")
        gmax = grp[col].transform("max")
        denom = (gmax - gmin).replace(0.0, 1.0)
        out[col] = ((out[col] - gmin) / denom).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# LGBM ranker training + ranking metrics
# ---------------------------------------------------------------------------

def hit_at(ranks: np.ndarray, k: int) -> float:
    return float((ranks < k).mean())


def ndcg_at(ranks: np.ndarray, k: int) -> float:
    gain = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
    return float(gain.mean())


def ranks_from_grouped_scores(scores: np.ndarray, group_size: int = 100) -> np.ndarray:
    """Reshape scores into (G, group_size), sort descending per row, return
    the rank of column 0 (the positive) per group. Lower rank = better."""
    G = scores.size // group_size
    s = scores.reshape(G, group_size)
    # rank of column 0 = number of columns with score strictly greater + tie-broken by lower index
    pos_score = s[:, 0:1]
    # Strictly greater handles ties with positive in column 0 by ranking it first
    ranks = (s[:, 1:] > pos_score).sum(axis=1)
    return ranks


def train_meta_learner(val_df: pd.DataFrame, score_cols: list[str], seed: int,
                       group_size: int = 100):
    import lightgbm as lgb

    # Group structure: group_size candidates per user-pair eval row.
    # The order in val_df matches the loader iteration order, so groups are
    # guaranteed contiguous. We sanity-check that in an assert.
    n = len(val_df)
    assert n % group_size == 0, (
        f"val rows {n} not a multiple of group_size {group_size}"
    )
    groups = np.full(n // group_size, group_size, dtype=np.int64)

    X = val_df[score_cols].values.astype(np.float32)
    y = val_df["label"].values.astype(np.int8)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=20,
        random_state=seed,
        verbosity=-1,
        eval_at=[5, 10, 20],
    )
    ranker.fit(X, y, group=groups)
    return ranker


def main():
    parser = argparse.ArgumentParser(description="Phase 3 meta-ensemble")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-negatives", type=int, default=99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/phase3_meta")
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"Device: {device}")

    kcore_dir = os.path.join("data", "processed", f"{args.kcore}core")
    n_users, n_items = get_n_users_items(kcore_dir)
    print(f"Users: {n_users:,}  Items: {n_items:,}")

    val_df  = load_split(kcore_dir, "val")
    test_df = load_split(kcore_dir, "test")
    user_pos_all = get_user_positive_items(kcore_dir)

    # Shared 1-vs-99 candidate pool - same seed as every other variant.
    val_ds  = EvalInteractionDataset(val_df,  n_items, user_pos_all,
                                      args.eval_negatives, args.seed)
    test_ds = EvalInteractionDataset(test_df, n_items, user_pos_all,
                                      args.eval_negatives, args.seed)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    group_size = args.eval_negatives + 1   # 1 positive + N negatives per user
    print(f"\nGroup size for ranking eval: {group_size} (1 pos + {args.eval_negatives} neg)")

    print("\nLoading 4 base-model checkpoints...")
    t0 = time.time()
    scorers = {
        "sasrec":      _load_sasrec(kcore_dir, n_items, device, args.seed,
                                     group_size=group_size),
        "lightgcn_hg": _load_lightgcn_hg(kcore_dir, n_users, n_items, device),
        "neumf_attn":  _load_neumf(kcore_dir, n_users, n_items, device),
        "text_ncf_mt": _load_text_ncf_mt(n_users, n_items, device),
    }
    print(f"  loaded in {time.time() - t0:.0f}s")

    print("\nScoring val...")
    t0 = time.time()
    val_table  = build_score_table(val_loader,  scorers, device, "val-score")
    print(f"  done in {time.time() - t0:.0f}s  ({len(val_table):,} pairs)")

    print("\nScoring test...")
    t0 = time.time()
    test_table = build_score_table(test_loader, scorers, device, "test-score")
    print(f"  done in {time.time() - t0:.0f}s  ({len(test_table):,} pairs)")

    print("\nApplying per-user min-max normalization...")
    val_table  = per_user_minmax(val_table,  MODEL_NAMES)
    test_table = per_user_minmax(test_table, MODEL_NAMES)

    print("\nTraining LGBMRanker on val...")
    t0 = time.time()
    ranker = train_meta_learner(val_table, MODEL_NAMES, args.seed,
                                 group_size=group_size)
    print(f"  done in {time.time() - t0:.0f}s")

    # ----- ranking metrics on test -----
    test_X = test_table[MODEL_NAMES].values.astype(np.float32)
    test_scores = ranker.predict(test_X)
    test_ranks  = ranks_from_grouped_scores(test_scores, group_size=group_size)

    test_metrics = {}
    for k in (5, 10, 20):
        test_metrics[f"HR@{k}"]   = hit_at(test_ranks, k)
        test_metrics[f"NDCG@{k}"] = ndcg_at(test_ranks, k)

    # ---- Per-base-model HR@10 / NDCG@10 on this same candidate pool ----
    # Sanity check: shows whether each model's score column ranks the
    # positive at column 0 above the 99 negatives. If SASRec's HR@10 here
    # is much higher than the meta-ensemble's, that's strong-model dilution.
    component_metrics = {}
    for col in MODEL_NAMES:
        ranks_col = ranks_from_grouped_scores(
            test_table[col].values.astype(np.float64), group_size=group_size)
        component_metrics[col] = {
            f"HR@{k}":   hit_at(ranks_col, k) for k in (5, 10, 20)
        } | {
            f"NDCG@{k}": ndcg_at(ranks_col, k) for k in (5, 10, 20)
        }

    # Best feature importances
    importances = dict(zip(MODEL_NAMES, ranker.feature_importances_.tolist()))

    print("\nMeta-ensemble test ranking metrics:")
    for m, v in sorted(test_metrics.items()):
        print(f"  {m}: {v:.4f}")
    print("\nPer-base-model HR@10 / NDCG@10 on the *same* seed-42 candidate pool:")
    for c in MODEL_NAMES:
        m = component_metrics[c]
        print(f"  {c:13s}  HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}")
    print("\nLightGBM feature importances (gain):")
    for n, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {n:13s}: {imp}")

    # ----- calibrated rating metrics (lstsq score → rating on val pos rows) -----
    # Only the positive (label=1) pairs have ground-truth ratings - lookup from
    # val_df + test_df by (user, item). Both splits have multiple rating rows
    # for the same (user, item) when users updated their reviews, so dedupe by
    # mean rating before the merge to preserve row count.
    def _ratings_for(pos_df: pd.DataFrame, split_df: pd.DataFrame) -> np.ndarray:
        rating_lookup = (split_df.groupby(["user_id", "item_id"], as_index=False)
                                  ["rating"].mean())
        joined = pos_df.merge(rating_lookup, on=["user_id", "item_id"], how="left")
        return joined["rating"].values.astype(np.float64)

    val_pos_mask  = val_table.label == 1
    test_pos_mask = test_table.label == 1
    val_pos_scores  = ranker.predict(val_table[val_pos_mask][MODEL_NAMES].values)
    test_pos_scores_arr = ranker.predict(test_table[test_pos_mask][MODEL_NAMES].values)

    val_pos_ratings  = _ratings_for(val_table[val_pos_mask],  val_df)
    test_pos_ratings = _ratings_for(test_table[test_pos_mask], test_df).astype(np.float32)
    assert len(val_pos_scores) == len(val_pos_ratings),  (len(val_pos_scores), len(val_pos_ratings))
    assert len(test_pos_scores_arr) == len(test_pos_ratings)

    # Drop any rows where the rating couldn't be looked up (shouldn't happen
    # but defensively guard).
    val_finite = np.isfinite(val_pos_ratings)
    val_pos_scores  = np.asarray(val_pos_scores)[val_finite]
    val_pos_ratings = val_pos_ratings[val_finite]

    vs = val_pos_scores.astype(np.float64)
    vy = val_pos_ratings.astype(np.float64)
    vs_std = float(np.std(vs))
    if vs_std > 1e-10:
        vs_n = (vs - vs.mean()) / vs_std
        A = np.vstack([vs_n, np.ones_like(vs_n)]).T
        coef, *_ = np.linalg.lstsq(A, vy, rcond=None)
        a = float(coef[0]) / vs_std
        b = float(coef[1]) - float(coef[0]) * vs.mean() / vs_std
    else:
        a, b = 0.0, float(vy.mean())

    pred = np.clip(a * test_pos_scores_arr + b, 1.0, 5.0).astype(np.float32)
    rating_metrics = {
        "rmse_calibrated": rmse_from_predictions(pred, test_pos_ratings),
        "mae_calibrated":  mae_from_predictions(pred, test_pos_ratings),
        "n":               int(len(test_pos_ratings)),
        "calibration_a":   float(a),
        "calibration_b":   float(b),
        "models":          MODEL_NAMES,
    }
    print(f"\nCalibrated rating: RMSE={rating_metrics['rmse_calibrated']:.4f}  "
          f"MAE={rating_metrics['mae_calibrated']:.4f}  "
          f"(a={a:.4f}, b={b:.4f})")

    # ----- save artefacts -----
    out_dir = args.out_dir
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(out_dir, "rating_metrics.json"), "w") as f:
        json.dump(rating_metrics, f, indent=2)
    with open(os.path.join(out_dir, "feature_importances.json"), "w") as f:
        json.dump(importances, f, indent=2)
    with open(os.path.join(out_dir, "component_metrics.json"), "w") as f:
        json.dump(component_metrics, f, indent=2)
    ranker.booster_.save_model(os.path.join(out_dir, "lightgbm.txt"))

    # Also save raw prediction scores for the notebook to inspect
    np.savez_compressed(
        os.path.join(out_dir, "test_scores.npz"),
        meta_scores=test_scores.astype(np.float32),
        component_scores=test_table[MODEL_NAMES].values.astype(np.float32),
        labels=test_table["label"].values.astype(np.int8),
        users=test_table["user_id"].values.astype(np.int64),
        items=test_table["item_id"].values.astype(np.int64),
    )

    print(f"\nSaved artefacts to {out_dir}/")


if __name__ == "__main__":
    main()
