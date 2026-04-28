"""One-shot patches to fix notebook cells the user flagged.

1. variants/hriday/notebooks/sasrec.ipynb cell 7 (Training curves):
   if the per-epoch CSV log isn't on disk, fall back to a textual
   summary built from the saved test_metrics_d128_L2.json instead of
   printing 'No SASRec logs yet'.

2. (No-op for now on neumf_attn — that notebook only needs a re-execute
   to pick up the MaxNLocator change that's already in the source.)

Idempotent: running this twice is safe.
"""
from __future__ import annotations
import nbformat
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SAS_NB = REPO / "variants" / "hriday" / "notebooks" / "sasrec.ipynb"

NEW_CELL_7 = '''import matplotlib.pyplot as plt

def load_json(p):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

# Try to plot the per-epoch metrics CSV first; fall back to a one-row
# summary table from the saved test_metrics if the log isn't available
# (logs/ is gitignored, so a fresh clone won't have the CSV).
curves = {}
if LOGS_SAS.exists():
    for csv in sorted(LOGS_SAS.glob("metrics_*.csv")):
        curves[csv.stem.replace("metrics_", "")] = pd.read_csv(csv)

if curves:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for name, df in curves.items():
        axes[0].plot(df["epoch"], df["HR@10"], marker="o", ms=3, label=name)
        axes[1].plot(df["epoch"], df["train_loss"], marker="o", ms=3, label=name)
    axes[0].set_title("Val HR@10"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    axes[1].set_title("Training loss (BPR)"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.show()
else:
    # Per-epoch CSV not present locally. Show the run summary derived from
    # the saved test_metrics JSON so the cell still produces output.
    test_meta = load_json(RES_SAS / "test_metrics_d128_L2.json")
    if test_meta is None:
        print("No SASRec results found.")
    else:
        summary = pd.DataFrame([{
            "config": f"dim={test_meta.get('embed_dim')}, L={test_meta.get('num_layers')}, heads={test_meta.get('num_heads')}",
            "max_seqlen": test_meta.get("max_seqlen"),
            "best_epoch": test_meta.get("best_epoch"),
            "best_val_HR@10": round(test_meta.get("best_val_HR@10", 0), 4),
            "test_HR@10": round(test_meta["HR@10"], 4),
            "test_NDCG@10": round(test_meta["NDCG@10"], 4),
            "train_time_s": round(test_meta.get("total_train_time_s", 0), 1),
        }]).T
        summary.columns = ["value"]
        print("SASRec full-scale run summary (per-epoch CSV not committed; logs/ is gitignored):")
        display(summary)
'''

nb = nbformat.read(SAS_NB, as_version=4)
# Find the training-curves cell. It's the one that begins with `import matplotlib.pyplot`
# and contains 'No SASRec logs yet' in its else branch.
patched = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    if "No SASRec logs yet" in c.source or ("LOGS_SAS.exists()" in c.source and "metrics_" in c.source):
        c.source = NEW_CELL_7.rstrip() + "\n"
        # Clear stale outputs so the next nbconvert --execute regenerates them
        c.outputs = []
        c.execution_count = None
        patched += 1

# Also clear outputs on cells 9 and 11 (the data-table cells) so they re-execute
# Their indexing is by section rather than ordinal, but cells with these starters:
RESET_PREFIXES = ("rows = []", "with open(RES_BASE")
for c in nb.cells:
    if c.cell_type == "code" and any(c.source.startswith(p) for p in RESET_PREFIXES):
        c.outputs = []
        c.execution_count = None
        patched += 1

nbformat.write(nb, SAS_NB)
print(f"patched {patched} cells in {SAS_NB.relative_to(REPO)}")

# ----------------------------------------------------------
# Also clear outputs on the neumf_attn training-curves cell so the
# refreshed MaxNLocator-based plot regenerates without the old crammed
# axis ticks.
# ----------------------------------------------------------
NEUMF_NB = REPO / "variants" / "aditya" / "notebooks" / "neumf_attn.ipynb"
nb2 = nbformat.read(NEUMF_NB, as_version=4)
patched2 = 0
for c in nb2.cells:
    if c.cell_type == "code" and "MaxNLocator" in c.source:
        c.outputs = []
        c.execution_count = None
        patched2 += 1

nbformat.write(nb2, NEUMF_NB)
print(f"patched {patched2} cells in {NEUMF_NB.relative_to(REPO)}")
