"""Repo-wide em-dash / en-dash -> ASCII hyphen sweep.

Replaces:
    U+2014 EM DASH       (—) -> -
    U+2013 EN DASH       (–) -> -
    U+2212 MINUS SIGN    (−) -> -

Walks the repo, skipping data/, .git/, results/*.npz, *.pkl, *.pt, and
the build_figures-cached PNGs. Edits .md, .py, .yaml/.yml, .sh,
.ipynb, .json, .txt, .csv files.

Idempotent. Run from repo root:
    python extras/dev_tooling/sanitize_dashes.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

REPLACEMENTS = {
    "\u2014": "-",   # —
    "\u2013": "-",   # –
    "\u2212": "-",   # − (rare, but matches when copying from rendered PDFs)
}

ALLOWED_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sh", ".ipynb", ".json",
                    ".txt", ".csv", ".rst"}

# Files / dirs to skip even if they have an allowed suffix.
SKIP_DIRS = {".git", "data", "__pycache__", ".venv", "venv", "node_modules"}
SKIP_FILES = {"sanitize_dashes.py"}  # don't rewrite this script's mapping table

def replace_in(text: str) -> str:
    out = text
    for k, v in REPLACEMENTS.items():
        out = out.replace(k, v)
    return out


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
    changed = 0
    for p in walk_files():
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = replace_in(text)
        if new != text:
            p.write_text(new, encoding="utf-8")
            rel = p.relative_to(REPO)
            print(f"sanitized {rel}")
            changed += 1
    print(f"\n{changed} files modified")


if __name__ == "__main__":
    main()
