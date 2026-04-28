"""Update every reference to the renamed (de-numbered) notebook filenames.

We dropped the 01_/02_/04_/05_/06_/07_/08_ prefixes from notebook
filenames so notebook lists don't suggest a fixed reading order. This
script rewrites every remaining `01_..._ipynb` → `..._ipynb` reference
in markdown, notebook markdown cells, and Python files.
"""
from __future__ import annotations
import os, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

RENAMES = {
    "01_dataset_eda.ipynb":         "dataset_eda.ipynb",
    "01_preprocessing.ipynb":       "preprocessing.ipynb",
    "02_baselines.ipynb":           "baselines.ipynb",
    "04_lightgcn_hg.ipynb":         "lightgcn_hg.ipynb",
    "05_ensemble_and_summary.ipynb":"ensemble_and_summary.ipynb",
    "06_sasrec.ipynb":              "sasrec.ipynb",
    "07_text_ncf.ipynb":            "text_ncf.ipynb",
    "08_neumf_attn.ipynb":          "neumf_attn.ipynb",
}

# Also rename notebook H1 titles ("# 06 - SASRec ..." -> "# SASRec ...")
TITLE_RENAMES = [
    (r"^(\\?\"|\")?# 0\d+ +-? *",  r"\1# "),     # leading "# 06 - " or "# 06 "
    (r"^(\\?\"|\")?## +0\d+\.\d* +-? *",  r"\1## "),  # "## 1.5 -" etc inside notebooks
    (r"^# 0\d+ +-? *",  "# "),                   # plain markdown
    (r"^## 0\d+\. +-? *",  "## "),               # plain markdown
]

ALLOWED_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sh", ".ipynb", ".json", ".txt"}
SKIP_DIRS = {".git", "data", "__pycache__", ".venv", "venv", "node_modules"}
SKIP_FILES = {"rename_notebook_refs.py"}


def walk_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f in SKIP_FILES:
                continue
            p = Path(root) / f
            if p.suffix.lower() in ALLOWED_SUFFIXES:
                yield p


def main():
    n_changed = 0
    for p in walk_files():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for k, v in RENAMES.items():
            new = new.replace(k, v)
        if new != text:
            p.write_text(new, encoding="utf-8")
            print(f"  refs updated: {p.relative_to(REPO)}")
            n_changed += 1
    print(f"\n{n_changed} files modified")


if __name__ == "__main__":
    main()
