"""
Extract hotel geographic metadata from TripAdvisor URLs in item2id.json.

Writes a small parquet under data/processed/hotel_meta/hotel_meta.parquet with one row per item in the 20-core subset. Does NOT touch data/processed/20core/.

TripAdvisor hotel URLs have the form: Hotel_Review-g{geo_id}-d{hotel_id}-Reviews-{name}-{location}.html

where {geo_id} is TripAdvisor's leaf-location identifier (1 per city/area) and {location} is an underscore-separated "City_Region[_Country]" slug whose last
1-2 tokens serve as a coarser regional pivot for hotels that are the only entry in their g_id.

Output columns
--------------
item_id         : int    -- matches the integer id used in train/val/test
hotel_url       : str    -- original TripAdvisor slug
g_id            : int    -- TripAdvisor location id (leaf)
hotel_name      : str    -- parsed from URL (before the first dash)
location_slug   : str    -- full underscore-separated suffix after the dash
region_slug     : str    -- last 2 tokens of location_slug (coarser pivot)
country_slug    : str    -- last 1 token of location_slug (coarsest pivot)

Usage
-----
    python -m scripts.extract_hotel_meta --kcore 20
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


URL_PATTERN = re.compile(
    r"^Hotel_Review-g(\d+)-d(\d+)-Reviews-(.+?)-(.+?)\.html$"
)


def parse_url(url: str):
    """Parse a TripAdvisor hotel-review URL.

    Returns (g_id, hotel_name, location_slug) or None if parse fails.
    """
    m = URL_PATTERN.match(url)
    if not m:
        return None
    g_id, _d_id, hotel_name, location_slug = m.groups()
    return int(g_id), hotel_name, location_slug


def region_and_country(location_slug: str) -> tuple[str, str]:
    """Coarser pivots derived from the location slug.

    Heuristic: TripAdvisor orders slug tokens from finest (city) to coarsest (region/country). So the last 2 tokens give a region-level pivot and the
    last 1 gives a country/state-level pivot. Not perfect across countries (e.g. "Ile_de_France" is one region but 3 tokens) but consistent enough
    to share pivots among nearby hotels. Falls back to the full slug when there are fewer tokens.
    """
    tokens = location_slug.split("_")
    if not tokens:
        return location_slug, location_slug
    country = tokens[-1]
    if len(tokens) >= 2:
        region = "_".join(tokens[-2:])
    else:
        region = country
    return region, country


def main():
    parser = argparse.ArgumentParser(description="Extract hotel metadata")
    parser.add_argument("--kcore", type=int, default=20)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed/hotel_meta")
    args = parser.parse_args()

    kcore_dir = Path(args.data_dir) / f"{args.kcore}core"
    item2id_path = kcore_dir / "item2id.json"
    if not item2id_path.exists():
        raise FileNotFoundError(f"Missing {item2id_path}")

    with open(item2id_path) as f:
        item2id = json.load(f)
    print(f"Loaded {len(item2id):,} hotel URLs from {item2id_path}")

    rows = []
    fail = []
    for url, iid in item2id.items():
        parsed = parse_url(url)
        if parsed is None:
            fail.append(url)
            continue
        g_id, hotel_name, location_slug = parsed
        region_slug, country_slug = region_and_country(location_slug)
        rows.append({
            "item_id": int(iid),
            "hotel_url": url,
            "g_id": g_id,
            "hotel_name": hotel_name,
            "location_slug": location_slug,
            "region_slug": region_slug,
            "country_slug": country_slug,
        })

    if fail:
        print(f"WARNING: {len(fail)} URLs failed to parse:")
        for u in fail[:5]:
            print(f"    {u}")

    df = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)

    # Sanity: item_id should be 0..n-1 contiguous (the processed data invariant)
    expected = set(range(len(item2id)))
    missing = expected - set(df["item_id"])
    if missing:
        raise RuntimeError(f"Missing item_ids: {sorted(list(missing))[:10]}...")

    # Summary stats
    stats = {
        "n_hotels": int(len(df)),
        "n_parse_failures": int(len(fail)),
        "n_unique_g_id": int(df["g_id"].nunique()),
        "n_unique_region": int(df["region_slug"].nunique()),
        "n_unique_country": int(df["country_slug"].nunique()),
        "g_id_singletons": int((df.groupby("g_id").size() == 1).sum()),
        "region_singletons": int((df.groupby("region_slug").size() == 1).sum()),
        "country_singletons": int((df.groupby("country_slug").size() == 1).sum()),
        "max_hotels_per_g_id": int(df.groupby("g_id").size().max()),
        "max_hotels_per_region": int(df.groupby("region_slug").size().max()),
        "max_hotels_per_country": int(df.groupby("country_slug").size().max()),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "hotel_meta.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df):,} rows to {parquet_path}")

    stats_path = out_dir / "hotel_meta_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    print()
    print("Metadata coverage:")
    print(f"  hotels              : {stats['n_hotels']:>8,}")
    print(f"  unique g_ids        : {stats['n_unique_g_id']:>8,}  "
          f"(singletons: {stats['g_id_singletons']:,})")
    print(f"  unique regions      : {stats['n_unique_region']:>8,}  "
          f"(singletons: {stats['region_singletons']:,})")
    print(f"  unique countries    : {stats['n_unique_country']:>8,}  "
          f"(singletons: {stats['country_singletons']:,})")


if __name__ == "__main__":
    main()
