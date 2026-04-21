"""
Summarize the LightGCN layer sweep + any extended runs.

Reads results/lightgcn/test_metrics_L*.json (including _d64 / _d128 variants)
and writes a markdown summary to results/lightgcn/summary.md with:
  - the K sweep at the base dim
  - extended-run comparisons
  - baseline vs best-LightGCN side-by-side
  - RMSE/MAE comparison (if rating metrics exist)

Usage: python scripts/summarize_lightgcn.py
"""

import glob
import json
import os
import re
from pathlib import Path

RESULTS = Path("results/lightgcn")
BASELINES = Path("results/baselines/baseline_results_20core.json")
RATING_BASELINES = Path("results/baselines/rating_metrics_20core.json")
RATING_LIGHTGCN = Path("results/lightgcn/rating_metrics_L1.json")
RATING_GMF = Path("results/gmf/rating_metrics.json")
ENSEMBLE = Path("results/lightgcn/ensemble_test_metrics.json")
OUT = RESULTS / "summary.md"


def load_runs() -> list[dict]:
    runs = []
    for p in sorted(glob.glob(str(RESULTS / "test_metrics_L*.json"))):
        with open(p) as f:
            m = json.load(f)
        m["_file"] = os.path.basename(p)
        # try to parse dim suffix from filename
        match = re.search(r"L(\d+)(?:_d(\d+))?\.json$", m["_file"])
        if match:
            m.setdefault("num_layers", int(match.group(1)))
            if match.group(2):
                m.setdefault("embed_dim", int(match.group(2)))
        runs.append(m)
    runs.sort(key=lambda r: (r.get("num_layers", 0), r.get("embed_dim", 0)))
    return runs


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt(v, p=4):
    if v is None:
        return "--"
    return f"{float(v):.{p}f}"


def main() -> None:
    runs = load_runs()
    if not runs:
        print(f"No runs found under {RESULTS}")
        return

    baselines = load_json(BASELINES)
    rating_baselines = load_json(RATING_BASELINES)
    rating_lg = load_json(RATING_LIGHTGCN)
    rating_gmf = load_json(RATING_GMF)
    ensemble = load_json(ENSEMBLE)

    lines = []
    lines.append("# LightGCN Results Summary\n")
    lines.append("20-core HotelRec subset: 46,660 users, 27,197 items, 1.85M "
                 "interactions (1.48M train / 184K val / 184K test).\n")

    # ---- K-sweep table ----
    lines.append("## K (layer count) sweep, dim=64, 30 epochs\n")
    lines.append("| K | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    sweep_runs = [r for r in runs if r.get("embed_dim", 64) == 64]
    best_sweep = max(sweep_runs, key=lambda r: r.get("HR@10", 0.0)) if sweep_runs else None
    for r in sweep_runs:
        bold = lambda v: f"**{fmt(v)}**" if r is best_sweep else fmt(v)
        lines.append(
            f"| {r.get('num_layers')} | {bold(r.get('HR@5'))} | {bold(r.get('HR@10'))} "
            f"| {bold(r.get('HR@20'))} | {bold(r.get('NDCG@5'))} | {bold(r.get('NDCG@10'))} "
            f"| {bold(r.get('NDCG@20'))} | {r.get('total_train_time_s','?')} |"
        )
    lines.append("")
    if best_sweep is not None:
        lines.append(
            f"**Best sweep config**: K = {best_sweep.get('num_layers')} (dim=64). "
            "Clean monotonic decline with K -- the 20-core graph is dense enough "
            "that direct neighbors dominate and deeper propagation over-smooths.\n"
        )

    # ---- Extended / follow-up runs ----
    ext_runs = [r for r in runs if r.get("embed_dim", 64) != 64]
    if ext_runs:
        lines.append("## Extended runs (larger embedding dim / more epochs)\n")
        lines.append("| K | dim | epochs | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | time (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in ext_runs:
            lines.append(
                f"| {r.get('num_layers')} | {r.get('embed_dim')} | {r.get('best_epoch','?')} "
                f"| {fmt(r.get('HR@5'))} | {fmt(r.get('HR@10'))} | {fmt(r.get('HR@20'))} "
                f"| {fmt(r.get('NDCG@5'))} | {fmt(r.get('NDCG@10'))} | {fmt(r.get('NDCG@20'))} "
                f"| {r.get('total_train_time_s','?')} |"
            )
        lines.append("")

    # ---- LightGCN best vs baselines ----
    all_runs_by_hr10 = sorted(runs, key=lambda r: r.get("HR@10", 0.0), reverse=True)
    best_lg = all_runs_by_hr10[0] if all_runs_by_hr10 else None
    if baselines and best_lg is not None:
        lines.append("## Ranking comparison: LightGCN vs Phase-1 baselines (test set)\n")
        lines.append("| Method | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in ("Popularity", "ItemKNN", "GMF"):
            b = baselines.get(name, {})
            lines.append(
                f"| {name} | {fmt(b.get('HR@5'))} | {fmt(b.get('HR@10'))} | {fmt(b.get('HR@20'))} "
                f"| {fmt(b.get('NDCG@5'))} | {fmt(b.get('NDCG@10'))} | {fmt(b.get('NDCG@20'))} |"
            )
        lg_label = f"**LightGCN (K={best_lg.get('num_layers')}, dim={best_lg.get('embed_dim','?')})**"
        lines.append(
            f"| {lg_label} | **{fmt(best_lg.get('HR@5'))}** | **{fmt(best_lg.get('HR@10'))}** "
            f"| **{fmt(best_lg.get('HR@20'))}** | **{fmt(best_lg.get('NDCG@5'))}** "
            f"| **{fmt(best_lg.get('NDCG@10'))}** | **{fmt(best_lg.get('NDCG@20'))}** |"
        )
        lines.append("")
        itemknn_hr10 = float(baselines.get("ItemKNN", {}).get("HR@10", 0.0) or 0.0)
        lg_hr10 = float(best_lg.get("HR@10", 0.0) or 0.0)
        if itemknn_hr10 > 0:
            pct = (lg_hr10 - itemknn_hr10) / itemknn_hr10 * 100.0
            lines.append(f"*LightGCN improves HR@10 over ItemKNN by **+{pct:.1f}%** relative.*\n")

    # ---- Ensemble with ItemKNN ----
    if ensemble:
        lines.append("## Ensemble: LightGCN + ItemKNN\n")
        best_w = ensemble.get("ensemble_best_w")
        tm = ensemble.get("test_metrics", {})
        lgcn = tm.get("lightgcn_only (w=1)", {})
        knn = tm.get("itemknn_only (w=0)", {})
        ens_key = next((k for k in tm if k.startswith("ensemble")), None)
        ens = tm.get(ens_key, {}) if ens_key else {}
        lines.append(
            f"Per-user min-max normalization then linear combination "
            f"`w * LightGCN + (1 - w) * ItemKNN`, with `w` tuned on the "
            f"validation split. Best `w = {best_w:.2f}`.\n"
        )
        lines.append("| Config | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(
            f"| ItemKNN (w=0) | {fmt(knn.get('HR@5'))} | {fmt(knn.get('HR@10'))} "
            f"| {fmt(knn.get('HR@20'))} | {fmt(knn.get('NDCG@5'))} "
            f"| {fmt(knn.get('NDCG@10'))} | {fmt(knn.get('NDCG@20'))} |"
        )
        lines.append(
            f"| LightGCN (w=1) | {fmt(lgcn.get('HR@5'))} | {fmt(lgcn.get('HR@10'))} "
            f"| {fmt(lgcn.get('HR@20'))} | {fmt(lgcn.get('NDCG@5'))} "
            f"| {fmt(lgcn.get('NDCG@10'))} | {fmt(lgcn.get('NDCG@20'))} |"
        )
        lines.append(
            f"| **Ensemble (w={best_w:.2f})** | **{fmt(ens.get('HR@5'))}** "
            f"| **{fmt(ens.get('HR@10'))}** | **{fmt(ens.get('HR@20'))}** "
            f"| **{fmt(ens.get('NDCG@5'))}** | **{fmt(ens.get('NDCG@10'))}** "
            f"| **{fmt(ens.get('NDCG@20'))}** |"
        )
        lines.append("")
        lines.append(
            "The ensemble improves every metric over pure LightGCN, but the "
            "magnitude is small (typically +0.001 to +0.005). The 10% "
            "ItemKNN weight is not enough to recover ItemKNN's top-1 "
            "concentration (NDCG@5 = 0.535 vs ItemKNN's 0.608), so the "
            "result is a modest refinement rather than a transformative "
            "combination. LightGCN already captures most of the exploitable "
            "collaborative signal on the 20-core graph.\n"
        )

    # ---- RMSE table (traditional metric) ----
    if rating_baselines:
        lines.append("## Rating-prediction comparison (test RMSE / MAE)\n")
        lines.append("RMSE and MAE on the held-out test set. Lower is better. "
                     "Note that ItemKNN and Popularity predict ratings natively "
                     "(weighted neighbor ratings / item mean rating), while "
                     "LightGCN is trained with BPR (pure ranking loss) -- its "
                     "RMSE is via linear calibration `rating = a*score + b` fit "
                     "on the validation split.\n")
        lines.append("| Method | RMSE | MAE | Notes |")
        lines.append("|---|---|---|---|")
        for name in ("GlobalMean", "Popularity", "ItemKNN"):
            m = rating_baselines.get(name, {})
            k_knn = rating_baselines.get("ItemKNN", {}).get("k_neighbors", "?")
            note = {
                "GlobalMean": "constant prediction, sanity baseline",
                "Popularity": "item-level mean rating from training",
                "ItemKNN":    f"weighted mean over top-{k_knn} neighbor ratings (dedup train)",
            }.get(name, "")
            lines.append(f"| {name} | {fmt(m.get('rmse'))} | {fmt(m.get('mae'))} | {note} |")
        if rating_gmf:
            lines.append(
                f"| GMF (dim={rating_gmf.get('embed_dim','?')}) "
                f"| {fmt(rating_gmf.get('rmse_calibrated'))} "
                f"| {fmt(rating_gmf.get('mae_calibrated'))} "
                f"| calibrated (a={fmt(rating_gmf.get('calibration_a'), p=4)}, "
                f"b={fmt(rating_gmf.get('calibration_b'), p=4)}) |"
            )
        if rating_lg:
            lines.append(
                f"| LightGCN (K={rating_lg.get('num_layers','?')}, dim={rating_lg.get('embed_dim','?')}) "
                f"| {fmt(rating_lg.get('rmse_calibrated'))} "
                f"| {fmt(rating_lg.get('mae_calibrated'))} "
                f"| calibrated (a={fmt(rating_lg.get('calibration_a'), p=4)}, "
                f"b={fmt(rating_lg.get('calibration_b'), p=4)}) |"
            )
        # Ensemble rating (from ensemble_eval.py's test_rating_metrics)
        if ensemble and "test_rating_metrics" in ensemble:
            rr = ensemble["test_rating_metrics"]
            ens_key = next((k for k in rr if k.startswith("ensemble")), None)
            if ens_key:
                ens = rr[ens_key]
                lines.append(
                    f"| LightGCN + ItemKNN ensemble (w={ens.get('w',0):.2f}) "
                    f"| **{fmt(ens.get('rmse'))}** | **{fmt(ens.get('mae'))}** "
                    f"| weighted combo of LightGCN-calibrated + ItemKNN-native rating preds |"
                )
        lines.append("")
        cal_a = rating_lg.get("calibration_a", 0.0) if rating_lg else 0.0
        lines.append(
            "**Interpretation**: Popularity wins RMSE because 78% of HotelRec "
            "ratings are 4-5 stars and the item-level mean captures most of the "
            "variance. ItemKNN's weighted neighbor formula tends to overshoot "
            "toward each item's own mean, losing user personalization at the "
            f"rating level. LightGCN's calibration slope is near zero "
            f"(`a = {fmt(cal_a)}`), confirming that BPR-trained embeddings do "
            "not carry calibrated rating information -- they are pure relevance "
            "scorers. This is expected and is why the shared evaluation framework "
            "uses ranking metrics as the primary comparison.\n"
        )

    # ---- Provenance ----
    lines.append("## Run metadata\n")
    for r in runs:
        lines.append(
            f"- `{r['_file']}`: K={r.get('num_layers')}, "
            f"dim={r.get('embed_dim','?')}, "
            f"best_val_HR@10={fmt(r.get('best_val_HR@10'))}, "
            f"best_epoch={r.get('best_epoch','?')}, "
            f"time={r.get('total_train_time_s','?')}s"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    if best_lg is not None:
        print(
            f"Best overall: K={best_lg.get('num_layers')}, dim={best_lg.get('embed_dim','?')} "
            f"-> HR@10={fmt(best_lg.get('HR@10'))}, NDCG@10={fmt(best_lg.get('NDCG@10'))}"
        )


if __name__ == "__main__":
    main()
