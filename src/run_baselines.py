"""
Run and compare all baselines on the same test set.

Usage:
    python -m src.run_baselines --kcore 20
"""

import argparse
import json
import os
import time

from src.data.dataset import get_dataloaders, get_n_users_items, load_split
from src.evaluation.ranking import evaluate_ranking
from src.models.popularity import PopularityBaseline
from src.models.knn import ItemKNN
from src.utils.seed import set_seed
from src.utils.io import load_config


def run_popularity(kcore_dir, k_values, batch_size=256, seed=42):
    """Run popularity baseline."""
    print("\n" + "=" * 60)
    print("  BASELINE: Popularity")
    print("=" * 60)

    n_users, n_items = get_n_users_items(kcore_dir)
    train_df = load_split(kcore_dir, "train")

    model = PopularityBaseline(n_items)
    model.fit(train_df)

    loaders = get_dataloaders(kcore_dir, batch_size=batch_size, seed=seed)

    start = time.time()
    results = evaluate_ranking(model, loaders["test"], k_values=k_values)
    elapsed = time.time() - start

    print(f"  Eval time: {elapsed:.1f}s")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v:.4f}")
    return results


def run_itemknn(kcore_dir, k_values, knn_k=20, batch_size=256, seed=42):
    """Run ItemKNN baseline."""
    print("\n" + "=" * 60)
    print(f"  BASELINE: ItemKNN (k={knn_k})")
    print("=" * 60)

    n_users, n_items = get_n_users_items(kcore_dir)
    train_df = load_split(kcore_dir, "train")

    model = ItemKNN(k=knn_k, n_users=n_users, n_items=n_items)
    model.fit(train_df)

    loaders = get_dataloaders(kcore_dir, batch_size=batch_size, seed=seed)

    start = time.time()
    results = evaluate_ranking(model, loaders["test"], k_values=k_values)
    elapsed = time.time() - start

    print(f"  Eval time: {elapsed:.1f}s")
    for k, v in sorted(results.items()):
        print(f"    {k}: {v:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run all baselines")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)

    data_cfg = load_config(args.data_config)
    kcore_dir = os.path.join(
        data_cfg["dataset"]["processed_dir"], f"{args.kcore}core"
    )
    if not os.path.isdir(kcore_dir):
        raise FileNotFoundError(
            f"Data not found: {kcore_dir}\n"
            f"Run: python -m src.data.preprocess --kcore {args.kcore}"
        )

    k_values = [5, 10, 20]
    all_results = {}

    all_results["Popularity"] = run_popularity(
        kcore_dir, k_values, args.batch_size, args.seed
    )
    all_results["ItemKNN"] = run_itemknn(
        kcore_dir, k_values, knn_k=20, batch_size=args.batch_size, seed=args.seed
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON")
    print("=" * 70)
    metrics = sorted(list(all_results["Popularity"].keys()))
    header = f"{'Model':<15}" + "".join(f"{m:>10}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        row = f"{name:<15}" + "".join(f"{res[m]:>10.4f}" for m in metrics)
        print(row)
    print("=" * 70)

    # Save
    results_dir = os.path.join("results", "baselines")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"baseline_results_{args.kcore}core.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
