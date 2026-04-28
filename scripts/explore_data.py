#!/usr/bin/env python3
"""
Explore the raw HotelRec dataset and print statistics.

Reads JSONL files from the raw data directory, computes counts, distributions, and coverage stats, then prints a formatted summary.

Uses streaming/online statistics (Welford's algorithm) so memory stays O(1) for all accumulators except user_review_counts and 
item_review_counts (which are needed for per-user/per-item stats and cannot be avoided).

Usage:
    python scripts/explore_data.py --data_dir data/raw --sample_size 0

    --sample_size 0   process ALL files (default)
    --sample_size 50  only process 50 files (quick sanity check)
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# streaming stats helper

class RunningStats:
    """Welford's online algorithm for mean and variance in O(1) memory."""

    __slots__ = ("n", "_mean", "_M2", "_min", "_max")

    def __init__(self):
        self.n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, x: float):
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._M2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def mean(self) -> float:
        return self._mean if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        return self._M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        return self._min if self.n > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self.n > 0 else 0.0


def _median_from_counter(counter: Counter) -> float:
    """Compute exact median from a Counter of {value: count}."""
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total == 0:
        return 0.0
    mid = total // 2
    cumulative = 0
    for val in sorted(counter.keys()):
        cumulative += counter[val]
        if total % 2 == 1:
            if cumulative > mid:
                return float(val)
        else:
            if cumulative == mid:
                next_val = min(k for k in counter if k > val)
                return (val + next_val) / 2.0
            elif cumulative > mid:
                return float(val)
    return 0.0


# helpers

SUB_RATING_KEYS = [
    "service", "cleanliness", "location", "value",
    "rooms", "sleep quality", "check-in", "business service",
]

# Display names for pretty printing
SUB_RATING_DISPLAY = {
    "service": "Service",
    "cleanliness": "Cleanliness",
    "location": "Location",
    "value": "Value",
    "rooms": "Rooms",
    "sleep quality": "Sleep Quality",
    "check-in": "Check-In",
    "business service": "Business Service",
}

# Review-length bucket boundaries (inclusive)
LENGTH_BUCKETS = [(1, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, None)]

# User-activity bucket boundaries
USER_ACTIVITY_BUCKETS = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, None)]


def _length_bucket_key(word_count: int) -> str:
    """Return the bucket label for a given word count."""
    for lo, hi in LENGTH_BUCKETS:
        if hi is None:
            if word_count >= lo:
                return f"{lo}+"
        elif lo <= word_count <= hi:
            return f"{lo}-{hi}"
    return "0"


def stream_file(fpath: str):
    """
    Stream reviews from a JSON file one at a time. Uses ijson for large files (>100MB) to avoid loading 10GB into RAM. Falls back to json.load for small files.
    """
    fsize = os.path.getsize(fpath)

    if fsize > 100_000_000:  # > 100MB → stream with ijson
        try:
            import ijson
            with open(fpath, "rb") as f:
                for review in ijson.items(f, "item"):
                    yield review
            return
        except ImportError:
            print("  WARNING: ijson not installed. pip install ijson")
            print("  Falling back to json.load - may OOM on large files.")

    # Small file or no ijson - load directly
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        first_char = ""
        for raw in f:
            first_char = raw.strip()[:1]
            if first_char:
                break

    if first_char == "[":
        # JSON array
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                for item in (data if isinstance(data, list) else [data]):
                    yield item
            except json.JSONDecodeError:
                return
    else:
        # JSONL (one object per line)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# main logic
def find_data_source(data_dir: str):
    """
    Find the data source. Could be:
    1. An archive (.tar.gz, .tar.bz2, .zip)
    2. JSON files (one or many)
    Returns (source_type, path_or_list)
    """
    import tarfile as _tarfile
    import zipfile as _zipfile

    data_path = Path(data_dir)

    # check for archives
    for pattern in ["*.tar.gz", "*.tar.bz2", "*.tgz", "*.zip"]:
        archives = list(data_path.glob(pattern))
        if archives:
            return ("archive", archives[0])

    # check for JSON files
    extensions = {".json", ".jsonl", ".txt"}
    json_files = sorted(
        p for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if json_files:
        return ("json_files", json_files)

    # check subdirectories
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            sub_jsons = sorted(subdir.glob("*.json"))
            if sub_jsons:
                return ("json_files", sub_jsons)

    return (None, None)


def stream_archive(archive_path):
    """Stream reviews from a zip/tar archive."""
    import io as _io
    import tarfile as _tarfile
    import zipfile as _zipfile

    path_str = str(archive_path)

    if path_str.endswith(".zip"):
        with _zipfile.ZipFile(path_str, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not (name.endswith(".json") or name.endswith(".txt")):
                    continue

                if info.file_size > 100_000_000:  # > 100MB → stream line-by-line (JSONL)
                    with zf.open(name) as f:
                        text_stream = _io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        for line in text_stream:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue
                    continue

                try:
                    content = zf.read(name).decode("utf-8", errors="replace")
                    data = json.loads(content)
                    if isinstance(data, list):
                        yield from data
                    elif isinstance(data, dict):
                        yield data
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    else:
        # tar handling - same fix for .txt
        with _tarfile.open(path_str, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not (member.name.endswith(".json") or member.name.endswith(".txt")):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                try:
                    if member.size > 100_000_000:  # > 100MB → stream line-by-line (JSONL)
                        text_stream = _io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        for line in text_stream:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue
                    else:
                        content = f.read().decode("utf-8", errors="replace")
                        data = json.loads(content)
                        if isinstance(data, list):
                            yield from data
                        elif isinstance(data, dict):
                            yield data
                except Exception:
                    continue
                finally:
                    f.close()


def explore(data_dir: str, sample_size: int = 0):
    data_path = Path(data_dir)

    source_type, source = find_data_source(data_dir)

    if source_type is None:
        print(f"No data found in {data_dir}")
        print("Download the dataset first:")
        print("  bash scripts/download_data.sh full     # full dataset (~10GB)")
        print("  bash scripts/download_data.sh sample   # synthetic sample for testing")
        sys.exit(1)

    total_files = 0
    if source_type == "archive":
        print(f"Found archive: {source.name} ({source.stat().st_size / 1e9:.1f} GB)")
        print("Streaming reviews directly from archive (no extraction needed)")
        review_stream = stream_archive(source)
        n_files = 1 
    else:
        all_files = source
        total_files = len(all_files)
        if sample_size > 0:
            all_files = all_files[:sample_size]
        n_files = len(all_files)
        print(f"Found {total_files} data files in {data_dir}")
        if sample_size > 0:
            print(f"Processing first {n_files} files (--sample_size {sample_size})")
        review_stream = None  # will iterate files below
    print()

    # Streaming accumulators (O(1) memory except Counters)

    total_reviews = 0

    # Rating histogram (ratings are integers 1-5, so exact histogram)
    rating_hist = Counter()          # {1: N, 2: N, ...}
    rating_stats = RunningStats()

    # Review-length bucketed counts + running stats
    length_bucket_counts = Counter()  # {"1-10": N, "11-50": N, ...}
    length_stats = RunningStats()
    n_with_text = 0

    # Temporal: year → count (extracted inline)
    year_counts = Counter()

    # Sub-ratings: running stats per key (O(1) per key)
    sub_rating_counts = {k: 0 for k in SUB_RATING_KEYS}
    sub_rating_stats = {k: RunningStats() for k in SUB_RATING_KEYS}

    # User/item Counters (unavoidable - needed for per-user/per-item stats)
    user_review_counts = Counter()
    item_review_counts = Counter()

    # Build a unified review iterator
    if review_stream is not None:
        # archive mode - already have a generator
        rec_iter = review_stream
    else:
        # file mode - chain all files
        def _file_iter():
            for idx, fpath in enumerate(all_files):
                if (idx + 1) % 500 == 0 or idx == 0:
                    print(f"  [{idx + 1}/{n_files}] Processing {fpath.name}...")
                yield from stream_file(str(fpath))
        rec_iter = _file_iter()

    for rec in rec_iter:
        total_reviews += 1
        if total_reviews % 1_000_000 == 0:
            print(f"    {total_reviews:,} reviews processed...")

        # User / item identifiers
        user = rec.get("author") or rec.get("user_url") or rec.get("user_id", "")
        item = rec.get("hotel_url") or rec.get("item_id", "")
        if user:
            user_review_counts[user] += 1
        if item:
            item_review_counts[item] += 1

        # Overall rating
        rating = rec.get("rating") or rec.get("overall_rating")
        if rating is not None:
            try:
                r = float(rating)
                rating_stats.update(r)
                rating_hist[round(r)] += 1
            except (ValueError, TypeError):
                pass

        # Review text length
        text = rec.get("text", "")
        if text:
            word_count = len(text.split())
            length_stats.update(word_count)
            n_with_text += 1
            bucket = _length_bucket_key(word_count)
            length_bucket_counts[bucket] += 1

        # Date - extract year inline instead of storing the string
        date_val = rec.get("date", "")
        if date_val:
            d_str = str(date_val).strip()
            for fmt_try in [d_str[:4], d_str[-4:]]:
                try:
                    year = int(fmt_try)
                    if 1990 <= year <= 2030:
                        year_counts[year] += 1
                        break
                except ValueError:
                    continue

        # Sub-ratings - could be in property_dict or top-level
        prop = rec.get("property_dict", {}) or {}
        for key in SUB_RATING_KEYS:
            # Check property_dict first, then top-level (with underscore variant)
            val = prop.get(key) or prop.get(key.replace("-", " "))
            if val is None:
                val = rec.get(key) or rec.get(key.replace(" ", "_"))
            if val is not None:
                try:
                    sub_rating_stats[key].update(float(val))
                    sub_rating_counts[key] += 1
                except (ValueError, TypeError):
                    pass

    # compute derived stats

    n_users = len(user_review_counts)
    n_items = len(item_review_counts)
    n_interactions = total_reviews

    if n_users > 0 and n_items > 0:
        density = n_interactions / (n_users * n_items) * 100
        sparsity = 100.0 - density
    else:
        density = 0.0
        sparsity = 100.0

    avg_per_user = n_interactions / n_users if n_users else 0
    avg_per_item = n_interactions / n_items if n_items else 0

    # User activity distribution - build a Counter of {count: n_users}
    user_activity_hist = Counter()
    users_1_review = 0
    users_lt5 = 0
    max_user_reviews = 0
    for c in user_review_counts.values():
        user_activity_hist[c] += 1
        if c == 1:
            users_1_review += 1
        if c < 5:
            users_lt5 += 1
        if c > max_user_reviews:
            max_user_reviews = c

    median_per_user = _median_from_counter(user_activity_hist)

    # Item popularity distribution - build a Counter of {count: n_items}
    item_popularity_hist = Counter()
    for c in item_review_counts.values():
        item_popularity_hist[c] += 1

    median_per_item = _median_from_counter(item_popularity_hist)

    # User activity buckets
    user_bucket_counts = {}
    for lo, hi in USER_ACTIVITY_BUCKETS:
        count = 0
        for c, n in user_activity_hist.items():
            if hi is None:
                if c >= lo:
                    count += n
            elif lo <= c <= hi:
                count += n
        if hi is None:
            user_bucket_counts[f"{lo}+"] = count
        else:
            user_bucket_counts[f"{lo}-{hi}"] = count

    # print results

    sep = "=" * 70
    print(f"\n{sep}")
    print("  HOTELREC DATASET - EXPLORATION SUMMARY")
    if sample_size > 0:
        print(f"  (based on {n_files}/{total_files} files)")
    print(sep)

    print(f"\n{'─' * 40}")
    print("  1. BASIC STATISTICS")
    print(f"{'─' * 40}")
    print(f"  Files processed:       {n_files:>15,}")
    print(f"  Total reviews:         {n_interactions:>15,}")
    print(f"  Unique users:          {n_users:>15,}")
    print(f"  Unique items (hotels): {n_items:>15,}")
    print(f"  Density:               {density:>14.5f}%")
    print(f"  Sparsity:              {sparsity:>14.5f}%")
    print(f"  Avg reviews/user:      {avg_per_user:>15.2f}")
    print(f"  Median reviews/user:   {median_per_user:>15.1f}")
    print(f"  Avg reviews/item:      {avg_per_item:>15.2f}")
    print(f"  Median reviews/item:   {median_per_item:>15.1f}")

    print(f"\n{'─' * 40}")
    print("  2. RATING DISTRIBUTION")
    print(f"{'─' * 40}")
    print(f"  Mean rating:           {rating_stats.mean:>15.2f}")
    print(f"  Median rating:         {_median_from_counter(rating_hist):>15.1f}")
    print(f"  Std rating:            {rating_stats.std:>15.2f}")
    print(f"  Min rating:            {rating_stats.min:>15.1f}")
    print(f"  Max rating:            {rating_stats.max:>15.1f}")
    print()
    print("  Rating histogram:")
    if rating_stats.n > 0:
        for star in [1, 2, 3, 4, 5]:
            count = rating_hist.get(star, 0)
            pct = count / rating_stats.n * 100
            bar = "█" * int(pct / 2)
            print(f"    {star}★: {count:>10,} ({pct:5.1f}%) {bar}")

    print(f"\n{'─' * 40}")
    print("  3. USER ACTIVITY DISTRIBUTION")
    print(f"{'─' * 40}")
    if n_users:
        print(f"  Users with exactly 1 review:  {users_1_review:>10,} ({users_1_review / n_users * 100:.2f}%)")
        print(f"  Users with < 5 reviews:       {users_lt5:>10,} ({users_lt5 / n_users * 100:.2f}%)")
    else:
        print("  N/A")
    print(f"  Max reviews by one user:      {max_user_reviews:>10,}")
    print()
    print("  Activity buckets:")
    for lo, hi in USER_ACTIVITY_BUCKETS:
        if hi is None:
            label = f"{lo}+"
            key = f"{lo}+"
        else:
            label = f"{lo}-{hi}"
            key = f"{lo}-{hi}"
        count = user_bucket_counts.get(key, 0)
        pct = count / n_users * 100 if n_users else 0
        print(f"    {label:>8s} reviews: {count:>10,} users ({pct:5.1f}%)")

    print(f"\n{'─' * 40}")
    print("  4. SUB-RATING COVERAGE")
    print(f"{'─' * 40}")
    for key in SUB_RATING_KEYS:
        cnt = sub_rating_counts[key]
        pct = cnt / n_interactions * 100 if n_interactions else 0
        stats = sub_rating_stats[key]
        avg_str = f"avg={stats.mean:.2f}" if stats.n > 0 else "avg=N/A"
        display = SUB_RATING_DISPLAY[key]
        print(f"    {display:<20s}: {cnt:>10,} ({pct:5.2f}%)  {avg_str}")

    print(f"\n{'─' * 40}")
    print("  5. REVIEW TEXT LENGTH (words)")
    print(f"{'─' * 40}")
    if n_with_text > 0:
        print(f"  Reviews with text:     {n_with_text:>15,} ({n_with_text / n_interactions * 100:.1f}%)")
        print(f"  Mean length:           {length_stats.mean:>15.2f}")
        print(f"  Std length:            {length_stats.std:>15.2f}")
        print(f"  Min length:            {length_stats.min:>15.0f}")
        print(f"  Max length:            {length_stats.max:>15.0f}")
        print()
        print("  Length buckets:")
        for lo, hi in LENGTH_BUCKETS:
            if hi is None:
                label = f"{lo}+"
                key = f"{lo}+"
            else:
                label = f"{lo}-{hi}"
                key = f"{lo}-{hi}"
            count = length_bucket_counts.get(key, 0)
            pct = count / n_with_text * 100
            print(f"    {label:>8s} words: {count:>10,} ({pct:5.1f}%)")
    else:
        print("  No review text found.")

    print(f"\n{'─' * 40}")
    print("  6. TEMPORAL DISTRIBUTION")
    print(f"{'─' * 40}")
    if year_counts:
        for year in sorted(year_counts.keys()):
            cnt = year_counts[year]
            pct = cnt / n_interactions * 100
            bar = "█" * max(1, int(pct))
            print(f"    {year}: {cnt:>10,} ({pct:5.1f}%) {bar}")
    else:
        print("  Could not parse dates.")

    # markdown table 

    print(f"\n{sep}")
    print("  COPY-PASTE TABLE FOR CHECKPOINT (Section 1a)")
    print(sep)
    print()
    print("| Stat | Value |")
    print("|------|-------|")
    print(f"| Number of users | {n_users:,} |")
    print(f"| Number of items | {n_items:,} |")
    print(f"| Number of interactions | {n_interactions:,} |")
    print(f"| Density | {density:.5f}% |")
    print(f"| Sparsity | {sparsity:.5f}% |")
    print(f"| Avg interactions per user | {avg_per_user:.2f} |")
    print(f"| Median interactions per user | {median_per_user:.1f} |")
    print(f"| Avg interactions per item | {avg_per_item:.2f} |")
    print(f"| Median interactions per item | {median_per_item:.1f} |")
    print(f"| Mean rating | {rating_stats.mean:.2f} ± {rating_stats.std:.2f} |")
    print(f"| Median rating | {_median_from_counter(rating_hist):.1f} |")
    print(f"| Mean review length (words) | {length_stats.mean:.2f} |")

    sub_parts = []
    for key in SUB_RATING_KEYS:
        pct = sub_rating_counts[key] / n_interactions * 100 if n_interactions else 0
        sub_parts.append(f"{SUB_RATING_DISPLAY[key]} ({pct:.1f}%)")
    print(f"| Sub-rating coverage | {', '.join(sub_parts)} |")
    print()

    print(f"{sep}")
    print("  Done.")
    print(sep)


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Explore the raw HotelRec dataset and print statistics "
                    "for the Day 6 checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files
  python scripts/explore_data.py --data_dir data/raw --sample_size 0

  # Quick test with 10 files
  python scripts/explore_data.py --data_dir data/raw --sample_size 10
        """,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to directory containing raw HotelRec JSON/JSONL files "
             "(default: data/raw)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of files to process. 0 = all files (default: 0)",
    )
    args = parser.parse_args()

    explore(args.data_dir, args.sample_size)


if __name__ == "__main__":
    main()
