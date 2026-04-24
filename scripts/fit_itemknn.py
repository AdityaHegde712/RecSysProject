"""Fit ItemKNN on the 20-core training split and pickle it to results/baselines/itemknn.pkl.

Needed by src/evaluate_ensemble.py and src/evaluate_two_stage.py, which expect
a pre-fit pickled ItemKNN.
"""

import argparse
import os
import sys
import time

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.dataset import get_n_users_items, load_split
from src.models.knn import ItemKNN
from src.utils.io import save_model


def main():
    parser = argparse.ArgumentParser(description="Fit ItemKNN and pickle it")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--k", type=int, default=20, help="neighbors per item")
    parser.add_argument("--out", default="results/baselines/itemknn.pkl")
    args = parser.parse_args()

    kcore_dir = os.path.join("data", "processed", f"{args.kcore}core")
    n_users, n_items = get_n_users_items(kcore_dir)
    train_df = load_split(kcore_dir, "train")
    print(f"{n_users:,} users, {n_items:,} items, {len(train_df):,} train rows")

    knn = ItemKNN(k=args.k, n_users=n_users, n_items=n_items)
    t0 = time.time()
    knn.fit(train_df, n_users=n_users, n_items=n_items)
    print(f"fit: {time.time() - t0:.0f}s")

    save_model(knn, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"saved {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
