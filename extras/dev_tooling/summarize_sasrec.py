"""
Summarize SASRec results.

SASRec is the primary variant under variants/hriday/. This script
aggregates the full-scale runs + calibrated RMSE into a single markdown
summary, plus a comparison vs Phase-1 baselines and the LightGCN-HG
secondary variant.

Reads:
    results/sasrec/test_metrics_d*_L*.json
    results/sasrec/rating_metrics_d*_L*.json
    results/lightgcn_hg/test_metrics_L1_d256_grc.json   (for HG comparison)
    results/baselines/baseline_results_20core.json       (baselines)

Writes:
    results/sasrec/summary.md

Usage:
    python scripts/summarize_sasrec.py
"""

import glob
import json
import os
import re
from pathlib import Path


SASREC_DIR = Path("results/sasrec")
HG_DIR     = Path("results/lightgcn_hg")
BASE       = Path("results/baselines/baseline_results_20core.json")
OUT        = SASREC_DIR / "summary.md"


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


def parse_fname(fname: str) -> tuple[int, int]:
    m = re.match(r"^test_metrics_d(\d+)_L(\d+)\.json$", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def load_runs() -> list[dict]:
    runs = []
    for p in sorted(glob.glob(str(SASREC_DIR / "test_metrics_d*_L*.json"))):
        m = load_json(Path(p))
        m["_file"] = os.path.basename(p)
        d, L = parse_fname(m["_file"])
        m.setdefault("embed_dim", d)
        m.setdefault("num_layers", L)
        runs.append(m)
    runs.sort(key=lambda r: (r.get("embed_dim", 0), r.get("num_layers", 0)))
    return runs


def load_rating_runs() -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(str(SASREC_DIR / "rating_metrics_d*_L*.json"))):
        fname = os.path.basename(p)
        mm = re.match(r"^rating_metrics_d(\d+)_L(\d+)\.json$", fname)
        if not mm:
            continue
        key = f"d{mm.group(1)}_L{mm.group(2)}"
        out[key] = load_json(Path(p))
    return out


def main() -> None:
    SASREC_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    rating_runs = load_rating_runs()
    baselines = load_json(BASE)
    hg = load_json(HG_DIR / "test_metrics_L1_d256_grc.json")

    lines = []
    lines.append("# SASRec Results Summary\n")
    lines.append(
        "Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018). "
        "Causal self-attention over per-user time-ordered hotel sequences, "
        "BPR loss, shared 1-vs-99 evaluation.\n"
    )
    lines.append("See `variants/hriday/PLAN.md` for the decision trail from "
                 "LightGCN-HG to SASRec as the primary variant.\n")

    if not runs:
        lines.append("> **No full-scale SASRec runs yet.** Run "
                     "`python -m src.train_sasrec --config configs/sasrec.yaml "
                     "--kcore 20` to populate.\n")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote empty-state summary to {OUT}")
        return

    # ---- Full-scale runs table ----
    lines.append("## Full-scale runs\n")
    lines.append("| dim | layers | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | best ep | time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append(
            f"| {r.get('embed_dim')} | {r.get('num_layers')} "
            f"| {fmt(r.get('HR@5'))} | {fmt(r.get('HR@10'))} | {fmt(r.get('HR@20'))} "
            f"| {fmt(r.get('NDCG@5'))} | {fmt(r.get('NDCG@10'))} | {fmt(r.get('NDCG@20'))} "
            f"| {r.get('best_epoch', '--')} | {fmt(r.get('total_train_time_s'), 1)} |"
        )
    lines.append("")

    best = max(runs, key=lambda r: r.get("HR@10", 0.0))

    # ---- SASRec vs baselines + LightGCN-HG ----
    if baselines:
        lines.append("## SASRec vs Phase-1 baselines and LightGCN-HG (secondary)\n")
        lines.append("| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in ("Popularity", "ItemKNN", "GMF"):
            b = baselines.get(name, {})
            lines.append(
                f"| {name} | {fmt(b.get('HR@5'))} | {fmt(b.get('HR@10'))} "
                f"| {fmt(b.get('HR@20'))} | {fmt(b.get('NDCG@5'))} "
                f"| {fmt(b.get('NDCG@10'))} | {fmt(b.get('NDCG@20'))} |"
            )
        if hg:
            lines.append(
                f"| LightGCN-HG (g_id+region+country) | {fmt(hg.get('HR@5'))} | "
                f"{fmt(hg.get('HR@10'))} | {fmt(hg.get('HR@20'))} | "
                f"{fmt(hg.get('NDCG@5'))} | {fmt(hg.get('NDCG@10'))} | "
                f"{fmt(hg.get('NDCG@20'))} |"
            )
        lines.append(
            f"| **SASRec (dim={best.get('embed_dim')}, L={best.get('num_layers')})** "
            f"| **{fmt(best.get('HR@5'))}** | **{fmt(best.get('HR@10'))}** "
            f"| **{fmt(best.get('HR@20'))}** | **{fmt(best.get('NDCG@5'))}** "
            f"| **{fmt(best.get('NDCG@10'))}** | **{fmt(best.get('NDCG@20'))}** |"
        )
        lines.append("")
        itemknn_hr10 = float(baselines.get("ItemKNN", {}).get("HR@10", 0.0))
        s_hr10 = float(best.get("HR@10", 0.0))
        itemknn_ndcg10 = float(baselines.get("ItemKNN", {}).get("NDCG@10", 0.0))
        s_ndcg10 = float(best.get("NDCG@10", 0.0))
        if itemknn_hr10 > 0:
            pct_hr = (s_hr10 - itemknn_hr10) / itemknn_hr10 * 100
            pct_n = (s_ndcg10 - itemknn_ndcg10) / itemknn_ndcg10 * 100
            lines.append(
                f"*SASRec improves HR@10 over ItemKNN by **+{pct_hr:.1f}%** "
                f"relative, NDCG@10 by **+{pct_n:.1f}%**.*\n"
            )

    # ---- RMSE ----
    if rating_runs:
        lines.append("## Rating prediction (calibrated)\n")
        lines.append("BPR doesn't output ratings natively; we fit "
                     "`rating = a * score + b` on val, evaluate on test. "
                     "The slope ends up near zero, so RMSE ≈ GlobalMean "
                     "(0.93). Popularity still wins RMSE on HotelRec "
                     "because 78 %% of ratings are 4-5 stars.\n")
        lines.append("| dim / L | RMSE | MAE | a | b | note |")
        lines.append("|---|---|---|---|---|---|")
        for key, r in rating_runs.items():
            note = r.get("fallback_note", "")
            lines.append(
                f"| {key} | {fmt(r.get('rmse_calibrated'))} | "
                f"{fmt(r.get('mae_calibrated'))} | {fmt(r.get('calibration_a'))} | "
                f"{fmt(r.get('calibration_b'))} | {note} |"
            )
        lines.append("")

    lines.append("## Run metadata\n")
    for r in runs:
        lines.append(
            f"- `{r['_file']}`: dim={r.get('embed_dim')}, "
            f"L={r.get('num_layers')}, heads={r.get('num_heads', '?')}, "
            f"seqlen={r.get('max_seqlen', '?')}, best_ep={r.get('best_epoch', '?')}, "
            f"time={r.get('total_train_time_s', '?')}s"
        )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Best: dim={best.get('embed_dim')} L={best.get('num_layers')}  "
          f"HR@10={fmt(best.get('HR@10'))}  NDCG@10={fmt(best.get('NDCG@10'))}")


if __name__ == "__main__":
    main()
