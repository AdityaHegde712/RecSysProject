#!/usr/bin/env python3
"""
Encode HotelRec review text into sentence embeddings.

Runs all-MiniLM-L6-v2 over the processed reviews and averages per user / per item. Saves as .npy files that TextNCF loads at train time.

Usage:
    python scripts/encode_text.py --kcore 20
    python scripts/encode_text.py --kcore 20 --device cpu
"""

import argparse
import os
import sys

# add project root to path so we can import src.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.text_embeddings import encode_reviews


def main():
    parser = argparse.ArgumentParser(description="Encode review text → embeddings")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/processed/text_emb")
    args = parser.parse_args()

    kcore_dir = os.path.join(args.data_dir, f"{args.kcore}core")
    if not os.path.isdir(kcore_dir):
        print(f"No processed data at {kcore_dir}")
        print(f"Run preprocessing first: python -m src.data.preprocess --kcore {args.kcore}")
        return

    encode_reviews(
        kcore_dir=kcore_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
