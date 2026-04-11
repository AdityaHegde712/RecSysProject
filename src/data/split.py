"""
Split processed HotelRec data into train/val/test sets.

Uses random splitting with a fixed seed for reproducibility.
Reads the processed parquet and writes three separate parquet files.

Usage:
    python -m src.data.split --kcore 20 --config configs/data.yaml
"""

import argparse
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random 80/10/10 split with fixed seed."""
    seed = cfg["split"]["seed"]
    train_ratio = cfg["split"]["train"]
    val_ratio = cfg["split"]["val"]
    test_ratio = cfg["split"]["test"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1, got {train_ratio + val_ratio + test_ratio}"

    # first split: train vs (val + test)
    holdout_ratio = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_ratio, random_state=seed
    )

    # second split: val vs test (relative ratio within holdout)
    val_relative = val_ratio / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df, test_size=(1 - val_relative), random_state=seed
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser(description="Split HotelRec data")
    parser.add_argument("--kcore", type=int, default=None,
                        help="k-core value (overrides config default)")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.kcore or cfg["kcore"]["default"]
    processed_dir = cfg["dataset"]["processed_dir"]
    kcore_dir = os.path.join(processed_dir, f"{k}core")

    # load processed interactions
    parquet_path = os.path.join(kcore_dir, "interactions.parquet")
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Total interactions: {len(df):,}")

    # split
    train_df, val_df, test_df = split_data(df, cfg)

    # save
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(kcore_dir, f"{name}.parquet")
        split_df.to_parquet(out_path, index=False)
        print(f"  {name}: {len(split_df):,} interactions -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
