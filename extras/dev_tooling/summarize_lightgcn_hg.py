"""
Summarize LightGCN-HG results.

LightGCN-HG is the secondary variant; the primary is SASRec (see
scripts/summarize_sasrec.py). This script aggregates the three-tier HG
run and its calibrated RMSE into a single markdown summary.

Reads:
    results/lightgcn_hg/test_metrics_L*_d*_*.json
    results/lightgcn_hg/rating_metrics_L*_d*_*.json
    results/baselines/baseline_results_20core.json

Writes:
    results/lightgcn_hg/summary.md

Usage:
    python scripts/summarize_lightgcn_hg.py
"""

import glob
import json
import os
import re
from pathlib import Path


HG_DIR     = Path("results/lightgcn_hg")
BASELINES  = Path("results/baselines/baseline_results_20core.json")
OUT        = HG_DIR / "summary.md"

TIER_LABELS = {
    "none": "none (bipartite only)",
    "g":    "g_id",
    "gr":   "g_id, region",
    "grc":  "g_id, region, country",
}


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt(v, p=4):
    if v is None:
        return "--"
    try:
        return f"{float(v):.{p}f}"
    except (TypeError, ValueError):
        return "--"


def parse_filename(fname: str) -> tuple[int, int, str]:
    """Extract (K, dim, tier_suffix) from test_metrics_L{K}_d{dim}_{suffix}.json."""
    m = re.match(r"^test_metrics_L(\d+)_d(\d+)_(\w+)\.json$", fname)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load_hg_runs() -> list[dict]:
    runs = []
    for p in sorted(glob.glob(str(HG_DIR / "test_metrics_L*_d*_*.json"))):
        with open(p) as f:
            m = json.load(f)
        m["_file"] = os.path.basename(p)
        K, dim, tsuf = parse_filename(m["_file"])
        m.setdefault("num_layers", K)
        m.setdefault("embed_dim", dim)
        m["tier_suffix"] = tsuf
        runs.append(m)
    order = {"none": 0, "g": 1, "gr": 2, "grc": 3}
    runs.sort(key=lambda r: (r.get("num_layers", 0), r.get("embed_dim", 0),
                             order.get(r["tier_suffix"], 99)))
    return runs


def load_hg_rating_runs() -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(str(HG_DIR / "rating_metrics_L*_d*_*.json"))):
        with open(p) as f:
            m = json.load(f)
        fname = os.path.basename(p)
        mm = re.match(r"^rating_metrics_L\d+_d\d+_(\w+)\.json$", fname)
        if mm:
            out[mm.group(1)] = m
    return out


def main() -> None:
    runs = load_hg_runs()
    rating_runs = load_hg_rating_runs()
    baselines = load_json(BASELINES)

    lines: list[str] = []
    lines.append("# LightGCN-HG Results Summary\n")
    lines.append(
        "Metadata-augmented LightGCN variant (secondary). Extends the "
        "(user, item) bipartite graph from the LightGCN paper (He et al., "
        "SIGIR 2020) with TripAdvisor location nodes parsed from the hotel "
        "URL (g_id, region, country), giving hotels a way to share signal "
        "through geographic pivots rather than only through common "
        "reviewers. Same BPR loop as the standard LightGCN formulation; "
        "only the adjacency changes.\n"
    )
    lines.append("See `variants/hriday/PLAN.md` for the design doc and the "
                 "decision trail that led to SASRec as the primary variant. "
                 "Walkthrough in `variants/hriday/notebooks/lightgcn_hg.ipynb`.\n")

    if not runs:
        lines.append("> **No HG runs found yet** under `results/lightgcn_hg/`. "
                     "Summary will populate once training JSONs land.\n")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote empty-state summary to {OUT}")
        return

    # ---- Runs table ----
    lines.append("## Run\n")
    lines.append("K=1, dim=256, num_negatives=2, bpr_reg=1e-5, bs=8192, "
                 "cosine LR, patience=15.\n")
    lines.append("| Tiers | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | nodes | edges (dir) | best ep | time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        tier_label = TIER_LABELS.get(r["tier_suffix"], r["tier_suffix"])
        lines.append(
            f"| {tier_label} | {fmt(r.get('HR@5'))} | {fmt(r.get('HR@10'))} "
            f"| {fmt(r.get('HR@20'))} | {fmt(r.get('NDCG@5'))} "
            f"| {fmt(r.get('NDCG@10'))} | {fmt(r.get('NDCG@20'))} "
            f"| {r.get('n_total_nodes', '--'):,} | {r.get('n_edges_directed', '--'):,} "
            f"| {r.get('best_epoch', '--')} | {r.get('total_train_time_s', '--')} |"
        )
    lines.append("")

    # ---- Best HG vs baselines ----
    best_hg = max(runs, key=lambda r: r.get("HR@10", 0.0))
    if baselines:
        lines.append("## Ranking: LightGCN-HG vs Phase-1 baselines\n")
        lines.append("| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in ("Popularity", "ItemKNN", "GMF"):
            b = baselines.get(name, {})
            lines.append(
                f"| {name} | {fmt(b.get('HR@5'))} | {fmt(b.get('HR@10'))} "
                f"| {fmt(b.get('HR@20'))} | {fmt(b.get('NDCG@5'))} "
                f"| {fmt(b.get('NDCG@10'))} | {fmt(b.get('NDCG@20'))} |"
            )
        best_label = TIER_LABELS.get(best_hg["tier_suffix"], best_hg["tier_suffix"])
        lines.append(
            f"| **LightGCN-HG ({best_label})** "
            f"| **{fmt(best_hg.get('HR@5'))}** | **{fmt(best_hg.get('HR@10'))}** "
            f"| **{fmt(best_hg.get('HR@20'))}** | **{fmt(best_hg.get('NDCG@5'))}** "
            f"| **{fmt(best_hg.get('NDCG@10'))}** | **{fmt(best_hg.get('NDCG@20'))}** |"
        )
        lines.append("")
        itemknn_hr10 = float(baselines.get("ItemKNN", {}).get("HR@10", 0.0) or 0.0)
        hg_hr10 = float(best_hg.get("HR@10", 0.0) or 0.0)
        if itemknn_hr10 > 0:
            pct = (hg_hr10 - itemknn_hr10) / itemknn_hr10 * 100.0
            lines.append(
                f"*LightGCN-HG improves HR@10 over ItemKNN by **{pct:+.1f}%** relative.*\n"
            )

    # ---- RMSE ----
    if rating_runs:
        lines.append("## Rating prediction (calibrated RMSE / MAE)\n")
        lines.append("BPR doesn't output ratings, so we fit "
                     "`rating = a * score + b` on the val split and report "
                     "test RMSE / MAE. The slope ends up near zero, so RMSE "
                     "≈ GlobalMean (0.93). Popularity wins RMSE on HotelRec "
                     "because 78 %% of ratings are 4-5 stars and item-mean "
                     "is near-optimal.\n")
        lines.append("| Method | RMSE | MAE | a | b |")
        lines.append("|---|---|---|---|---|")
        for suf, r in rating_runs.items():
            tl = TIER_LABELS.get(suf, suf)
            lines.append(
                f"| LightGCN-HG ({tl}) "
                f"| {fmt(r.get('rmse_calibrated'))} "
                f"| {fmt(r.get('mae_calibrated'))} "
                f"| {fmt(r.get('calibration_a'))} "
                f"| {fmt(r.get('calibration_b'))} |"
            )
        lines.append("")

    # ---- Provenance ----
    lines.append("## Run metadata\n")
    for r in runs:
        lines.append(
            f"- `{r['_file']}`: tiers={r['tier_suffix']}, "
            f"K={r.get('num_layers')}, dim={r.get('embed_dim')}, "
            f"best_val_HR@10={fmt(r.get('best_val_HR@10'))}, "
            f"best_epoch={r.get('best_epoch', '?')}, "
            f"time={r.get('total_train_time_s', '?')}s, "
            f"nodes={r.get('n_total_nodes', '?')}, "
            f"edges_dir={r.get('n_edges_directed', '?')}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    if best_hg is not None:
        print(
            f"Best HG: tiers={best_hg['tier_suffix']}  "
            f"HR@10={fmt(best_hg.get('HR@10'))}  NDCG@10={fmt(best_hg.get('NDCG@10'))}"
        )


if __name__ == "__main__":
    main()
