# -*- coding: utf-8 -*-
"""Verify that all dependencies are installed and working.

Run this BEFORE submitting batch jobs to catch issues early:
    python scripts/verify_env.py

Checks every import used by the HotelRec codebase and reports
all failures at once instead of one at a time.
"""

import os
import sys
import importlib

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PASS = 0
FAIL = 0


def check(name, import_path=None, version_attr='__version__'):
    """Try importing a module and report success/failure."""
    global PASS, FAIL
    mod_name = import_path or name
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, version_attr, '?') if version_attr else '?'
        print(f"  OK   {name:<30s} {ver}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL {name:<30s} {e}")
        FAIL += 1


def check_optional(name, import_path=None, version_attr='__version__'):
    """Check an optional dependency — warn but don't fail."""
    global PASS
    mod_name = import_path or name
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, version_attr, '?') if version_attr else '?'
        print(f"  OK   {name:<30s} {ver}")
        PASS += 1
    except Exception:
        print(f"  WARN {name:<30s} not installed (optional — needed for neural variants)")
        PASS += 1  # don't count as failure


def check_src_imports():
    """Check that all project module imports work."""
    global PASS, FAIL
    modules = [
        'src',
        'src.models',
        'src.models.common',
        'src.models.knn',
        'src.data.dataset',
        'src.data.preprocess',
        'src.data.split',
        'src.evaluation.ranking',
        'src.utils.io',
        'src.utils.seed',
        'src.utils.metrics_logger',
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            print(f"  OK   {mod_name:<30s}")
            PASS += 1
        except Exception as e:
            print(f"  FAIL {mod_name:<30s} {e}")
            FAIL += 1


def check_script_syntax():
    """Check that training/eval scripts can be parsed."""
    global PASS, FAIL
    import py_compile
    scripts = [
        'src/run_baselines.py',
        'src/train_gmf.py',
    ]
    for script in scripts:
        try:
            py_compile.compile(script, doraise=True)
            print(f"  OK   {script:<30s} syntax ok")
            PASS += 1
        except Exception as e:
            print(f"  FAIL {script:<30s} {e}")
            FAIL += 1


if __name__ == '__main__':
    print("=" * 60)
    print("HotelRec — Environment Verification")
    print("=" * 60)
    print(f"\nPython: {sys.version}")
    print(f"Path:   {sys.executable}\n")

    print("--- Core dependencies (required) ---")
    check('numpy')
    check('scipy')
    check('pandas')
    check('scikit-learn', import_path='sklearn', version_attr='__version__')
    check('pyyaml', import_path='yaml', version_attr='__version__')
    check('tqdm')

    print("\n--- Optional dependencies ---")
    check_optional('torch')
    check_optional('matplotlib')
    check_optional('Pillow', import_path='PIL', version_attr='__version__')

    print("\n--- Project imports ---")
    check_src_imports()

    print("\n--- Script syntax ---")
    check_script_syntax()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("\nFix the failures above before submitting batch jobs.")
        print("Common fixes:")
        print("  pip install <package>")
        print("  pip install --only-binary=:all: <package>")
        sys.exit(1)
    else:
        print("\nAll checks passed! Ready to submit:")
        print("  python -m src.train --config configs/itemknn.yaml --kcore 20")
    print("=" * 60)
