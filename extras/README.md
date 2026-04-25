# extras/

Files that aren't on the end-to-end reproducibility path the top-level
README's Quick Start uses, kept around so nothing is lost.

- **`hpc/`** — Pramod's HPC convenience layer: SLURM submit script,
  shell aliases, conda-free env setup, and `requirements-hpc.txt`
  (a tighter pin set that handles GLIBC 2.17 on the SJSU COE cluster).
  The canonical setup is local pip + the top-level `requirements.txt`;
  the HPC layer is optional and Pramod-specific.

- **`dev_tooling/`** — auto-generators for the variant `summary.md`
  files (`summarize_sasrec.py`, `summarize_lightgcn_hg.py`). The shipped
  summaries were hand-edited after generation (vanilla-vs-enhanced
  tables, decision narrative), so re-running these would overwrite that
  work. Kept here for traceability.

Nothing under `extras/` is required to reproduce the project results.
