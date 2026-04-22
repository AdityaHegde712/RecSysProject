"""
Torch-free graph construction for LightGCN-HG.

Living under ``src/graph/`` (not ``src/models/``) so importing it doesn't
trigger ``src.models.__init__``, which eagerly imports every model and
therefore torch. This lets notebooks inspect graph structure in an
environment where torch can't load -- e.g. a kernel started while another
GPU process holds the CUDA DLL on Windows.

``src/models/lightgcn_hg.py`` imports and wraps these functions to
produce its torch sparse tensor.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_hg_norm_adj_scipy(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    hotel_meta: pd.DataFrame,
    n_users: int,
    n_items: int,
    tiers: Iterable[str] = ("g_id", "region_slug", "country_slug"),
) -> tuple[sp.coo_matrix, dict]:
    """Build the symmetrically-normalised hetero adjacency as scipy sparse.

    Nodes are laid out in one contiguous block: ``[users | items | tier_1 |
    tier_2 | ...]``. Each tier column in ``hotel_meta`` contributes its own
    contiguous range.

    Edges:
        * user <-> item (from the interaction arrays, both directions)
        * item <-> tier node (one edge per hotel per selected tier,
          both directions)

    Returns:
        A_hat : (N, N) scipy sparse COO -- ``D^{-1/2} A D^{-1/2}``.
        meta  : dict with node counts, offsets, and per-tier id maps.
    """
    # 1. Build integer id maps for each tier (contiguous 0..k-1 per tier).
    tier_maps: dict[str, dict] = {}
    tier_sizes: dict[str, int] = {}
    for col in tiers:
        if col not in hotel_meta.columns:
            raise KeyError(f"Tier column '{col}' missing from hotel_meta")
        uniq = sorted(hotel_meta[col].astype(str).unique())
        tier_maps[col] = {v: i for i, v in enumerate(uniq)}
        tier_sizes[col] = len(uniq)

    # 2. Lay out global node indices.
    offsets = {"user": 0, "item": n_users}
    cursor = n_users + n_items
    for col in tiers:
        offsets[col] = cursor
        cursor += tier_sizes[col]
    n_total = cursor

    # 3. COO-build A.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    # 3a. User-item edges (both directions).
    rows.append(user_ids.astype(np.int64))
    cols.append((item_ids + offsets["item"]).astype(np.int64))
    rows.append((item_ids + offsets["item"]).astype(np.int64))
    cols.append(user_ids.astype(np.int64))

    # 3b. Item-tier edges (both directions), one per hotel per tier.
    meta_sorted = hotel_meta.sort_values("item_id")
    item_global = (meta_sorted["item_id"].values + offsets["item"]).astype(np.int64)
    for col in tiers:
        tier_local = meta_sorted[col].astype(str).map(tier_maps[col]).values.astype(np.int64)
        tier_global = tier_local + offsets[col]
        rows.append(item_global)
        cols.append(tier_global)
        rows.append(tier_global)
        cols.append(item_global)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    data = np.ones(len(all_rows), dtype=np.float32)

    A = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n_total, n_total))
    deg = np.array(A.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_total": n_total,
        "tier_sizes": tier_sizes,
        "offsets": offsets,
        "tier_maps": tier_maps,
        "tiers": tuple(tiers),
        "n_edges_sym": int(len(all_rows)),
    }
    return A_hat, meta
