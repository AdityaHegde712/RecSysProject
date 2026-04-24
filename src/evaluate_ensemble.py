"""
Ensemble scoring: combine TextNCF, ItemKNN, and GMF predictions.

Min-max normalizes each model's scores per user, then does a grid search
over weight combinations (w1+w2+w3=1) on the validation set. Reports
the best ensemble on the test set.

Usage:
    python -m src.evaluate_ensemble \
        --text-ncf-ckpt results/text_ncf/best_model.pt \
        --gmf-ckpt results/gmf/best_model.pt \
        --knn-ckpt results/baselines/itemknn.pkl \
        --kcore 20

"""

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import get_dataloaders, get_n_users_items, load_split
from src.data.text_embeddings import load_text_embeddings, TEXT_EMB_DIR
from src.models.text_ncf import TextNCF
from src.models.gmf import GMF
from src.models.knn import ItemKNN
from src.utils.io import load_config, load_checkpoint, load_model
from src.utils.seed import set_seed


def _get_device():
    """Pick CUDA if available and working, else CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def _score_neural(model, users, items, device):
    """Score (user, item) pairs with a neural model. Returns (B, C) tensor."""
    batch_size, num_cands = items.shape
    u_flat = users.unsqueeze(1).expand(-1, num_cands).reshape(-1).to(device)
    i_flat = items.reshape(-1).to(device)
    with torch.no_grad():
        scores = model(u_flat, i_flat).reshape(batch_size, num_cands)
    return scores.cpu()


def _score_knn(knn_model, users, items):
    """Score with ItemKNN. Returns (B, C) tensor."""
    return knn_model.predict_batch(users, items)


def minmax_normalize(scores):
    """Per-row min-max normalization. scores: (B, C) tensor."""
    mins = scores.min(dim=1, keepdim=True).values
    maxs = scores.max(dim=1, keepdim=True).values
    denom = maxs - mins
    # avoid division by zero for constant rows
    denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
    return (scores - mins) / denom


def evaluate_ensemble_loader(models, loader, weights, device):
    """Run ensemble evaluation on a dataloader.

    models: dict with keys 'text_ncf', 'gmf', 'knn'
    weights: (w_text, w_gmf, w_knn)
    Returns dict of HR@k, NDCG@k metrics.
    """
    k_values = [5, 10, 20]
    metrics = {f"{m}@{k}": [] for k in k_values for m in ("HR", "NDCG")}

    w_text, w_gmf, w_knn = weights

    for users, items, labels in tqdm(loader, desc="ensemble-eval", leave=False):
        batch_size = users.size(0)

        # get raw scores from each model
        s_text = _score_neural(models["text_ncf"], users, items, device)
        s_gmf = _score_neural(models["gmf"], users, items, device)
        s_knn = _score_knn(models["knn"], users, items)

        # normalize per user
        s_text = minmax_normalize(s_text)
        s_gmf = minmax_normalize(s_gmf)
        s_knn = minmax_normalize(s_knn)

        # weighted combination
        combined = w_text * s_text + w_gmf * s_gmf + w_knn * s_knn

        _, indices = torch.sort(combined, dim=1, descending=True)

        for b in range(batch_size):
            ranked = items[b][indices[b]].tolist()
            gt_item = items[b, 0].item()

            for k in k_values:
                top_k = ranked[:k]
                # HR
                metrics[f"HR@{k}"].append(float(gt_item in top_k))
                # NDCG
                ndcg_val = 0.0
                for i, item in enumerate(top_k):
                    if item == gt_item:
                        ndcg_val = 1.0 / math.log2(i + 2)
                        break
                metrics[f"NDCG@{k}"].append(ndcg_val)

    return {key: float(np.mean(vals)) for key, vals in metrics.items()}


def grid_search_weights(models, val_loader, device, step=0.1):
    """Grid search over weight combinations that sum to 1.

    Returns (best_weights, best_metrics, all_results).
    """
    # generate weight combos: w1 + w2 + w3 = 1, step 0.1
    candidates = np.arange(0.0, 1.0 + step / 2, step)
    weight_combos = []
    for w1 in candidates:
        for w2 in candidates:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-6:
                w3 = max(w3, 0.0)
                weight_combos.append((round(w1, 2), round(w2, 2), round(w3, 2)))

    print(f"Grid search: {len(weight_combos)} weight combinations")

    best_ndcg = -1.0
    best_weights = (0.33, 0.33, 0.34)
    best_metrics = {}
    all_results = []

    for i, (w_text, w_gmf, w_knn) in enumerate(weight_combos):
        m = evaluate_ensemble_loader(models, val_loader, (w_text, w_gmf, w_knn), device)
        ndcg10 = m["NDCG@10"]
        all_results.append({
            "w_text": w_text, "w_gmf": w_gmf, "w_knn": w_knn,
            **m,
        })

        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            best_weights = (w_text, w_gmf, w_knn)
            best_metrics = m

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(weight_combos)}] best so far: "
                  f"w=({best_weights[0]:.1f},{best_weights[1]:.1f},{best_weights[2]:.1f}) "
                  f"NDCG@10={best_ndcg:.4f}")

    return best_weights, best_metrics, all_results


def load_text_ncf_model(ckpt_path, config_path, num_users, num_items, device):
    """Load a trained TextNCF model from checkpoint."""
    config = load_config(config_path)
    model_cfg = config.get("model", {})

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

    # load text embeddings
    emb_dir = config.get("paths", {}).get("text_emb_dir", TEXT_EMB_DIR)
    user_emb, item_emb = load_text_embeddings(emb_dir)
    model.set_text_embeddings(torch.from_numpy(user_emb), torch.from_numpy(item_emb))

    load_checkpoint(ckpt_path, model=model)
    model.to(device)
    model.eval()
    return model


def load_gmf_model(ckpt_path, config_path, num_users, num_items, device):
    """Load a trained GMF model from checkpoint."""
    config = load_config(config_path)
    embed_dim = config["model"].get("embedding_dim",
                                     config["model"].get("embed_dim", 64))
    model = GMF(num_users, num_items, embed_dim=embed_dim)
    load_checkpoint(ckpt_path, model=model)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument("--text-ncf-ckpt", default="results/text_ncf/best_model.pt")
    parser.add_argument("--text-ncf-config", default="configs/text_ncf.yaml")
    parser.add_argument("--gmf-ckpt", default="results/gmf/best_model.pt")
    parser.add_argument("--gmf-config", default="configs/gmf.yaml")
    parser.add_argument("--knn-ckpt", default="results/baselines/itemknn.pkl")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--grid-step", type=float, default=0.1,
                        help="Weight grid step size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = _get_device()
    print(f"Device: {device}")

    # load data
    kcore_dir = os.path.join("data", "processed", f"{args.kcore}core")
    if not os.path.isdir(kcore_dir):
        print(f"No data at {kcore_dir}. Run preprocessing first.")
        return

    num_users, num_items = get_n_users_items(kcore_dir)
    print(f"Dataset: {num_users:,} users, {num_items:,} items")

    loaders = get_dataloaders(kcore_dir, batch_size=256, eval_negatives=99,
                              seed=args.seed)

    # load models
    print("\nLoading models...")
    text_ncf = load_text_ncf_model(
        args.text_ncf_ckpt, args.text_ncf_config,
        num_users, num_items, device)
    print(f"  TextNCF: {text_ncf.count_parameters():,} params")

    gmf = load_gmf_model(
        args.gmf_ckpt, args.gmf_config,
        num_users, num_items, device)
    n_gmf = sum(p.numel() for p in gmf.parameters())
    print(f"  GMF: {n_gmf:,} params")

    knn = load_model(args.knn_ckpt)
    print(f"  ItemKNN: k={knn.k}, sim nnz={knn.sim.nnz:,}")

    models = {"text_ncf": text_ncf, "gmf": gmf, "knn": knn}

    # grid search on validation set
    print("\n" + "=" * 65)
    print("Grid search on validation set")
    print("=" * 65)
    best_weights, val_metrics, all_results = grid_search_weights(
        models, loaders["val"], device, step=args.grid_step)

    w_text, w_gmf, w_knn = best_weights
    print(f"\nBest weights: TextNCF={w_text:.2f}, GMF={w_gmf:.2f}, "
          f"ItemKNN={w_knn:.2f}")
    print(f"Val metrics: " + "  ".join(f"{k}={v:.4f}"
          for k, v in sorted(val_metrics.items())))

    # evaluate on test set with best weights
    print("\n" + "=" * 65)
    print("Test evaluation with best weights")
    print("=" * 65)
    test_metrics = evaluate_ensemble_loader(
        models, loaders["test"], best_weights, device)

    print("\nTest Results:")
    for m, v in sorted(test_metrics.items()):
        print(f"  {m}: {v:.4f}")

    # save results
    out_dir = "results/text_ncf"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "best_weights": {
            "text_ncf": w_text, "gmf": w_gmf, "knn": w_knn,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "grid_step": args.grid_step,
        "num_combos_searched": len(all_results),
    }

    out_path = os.path.join(out_dir, "ensemble_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # also save full grid search results for analysis
    grid_path = os.path.join(out_dir, "ensemble_grid_search.json")
    with open(grid_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved grid search details to {grid_path}")


if __name__ == "__main__":
    main()
