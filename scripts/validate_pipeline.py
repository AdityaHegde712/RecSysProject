# -*- coding: utf-8 -*-
"""Validate the HotelRec pipeline before submitting HPC jobs.

Tests the GMF model: creation, forward pass, checkpoint save/load,
ranking metrics, and metrics logging.

Usage:
    python scripts/validate_pipeline.py

Run this after setup to catch issues before wasting GPU hours.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def step(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f'  PASS  {name}')
        PASS += 1
    except Exception as e:
        print(f'  FAIL  {name}: {e}')
        FAIL += 1


# ─── Tests ────────────────────────────────────────────────────────────

def test_core_imports():
    import torch
    import numpy
    import scipy
    import pandas
    import sklearn
    import yaml
    import tqdm


def test_gpu_detection():
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f'         GPU: {gpu} ({mem / 1e9:.1f} GB)')
    else:
        print('         No GPU — skipping GPU tests (OK on login node)')


def test_model_creation():
    from src.models.common import build_model

    num_users, num_items = 100, 50
    cfg = {'model': {'name': 'gmf', 'embed_dim': 16}}
    model = build_model(cfg, num_users, num_items)
    assert model is not None, 'GMF returned None'
    n_params = sum(p.numel() for p in model.parameters())
    print(f'         gmf: {n_params:,} params')


def test_gmf_forward():
    import torch
    from src.models.common import build_model

    num_users, num_items = 100, 50
    batch_size = 8
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))

    cfg = {'model': {'name': 'gmf', 'embed_dim': 16}}
    model = build_model(cfg, num_users, num_items)
    out = model(user_ids, item_ids)
    assert out.shape == (batch_size,), f'expected ({batch_size},), got {out.shape}'
    assert out.min() >= 0 and out.max() <= 1, f'output not in [0,1]'
    print(f'         input ({batch_size},) -> output {out.shape}, range [{out.min():.3f}, {out.max():.3f}]')


def test_ranking_metrics():
    from src.metrics.ranking import hit_ratio, ndcg

    ranked = [5, 3, 1, 7, 2]
    gt = 3

    hr5 = hit_ratio(ranked, gt, k=5)
    ndcg5 = ndcg(ranked, gt, k=5)
    assert hr5 == 1.0, f'HR@5 should be 1.0, got {hr5}'
    assert ndcg5 > 0, f'NDCG@5 should be > 0, got {ndcg5}'
    print(f'         HR@5={hr5:.2f}, NDCG@5={ndcg5:.4f}')


def test_checkpoint_roundtrip():
    import torch
    from src.models.common import build_model
    from src.utils.io import save_checkpoint, load_checkpoint

    num_users, num_items = 50, 30
    cfg = {'model': {'name': 'gmf', 'embed_dim': 8}}
    model = build_model(cfg, num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
        save_checkpoint(model, optimizer, epoch=5, path=f.name)
        loaded_model, epoch = load_checkpoint(f.name, model)
        assert epoch == 5, f'expected epoch 5, got {epoch}'
        assert loaded_model is not None
    print(f'         save + load: epoch={epoch}')


def test_metrics_logger():
    from src.utils.metrics_logger import MetricsLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir, filename='test_metrics.csv')
        logger.log(1, {'loss': 0.5, 'HR@10': 0.3})
        logger.log(2, {'loss': 0.4, 'HR@10': 0.4})

        df = logger.load()
        assert len(df) == 2, f'expected 2 rows, got {len(df)}'
        assert df.iloc[0]['epoch'] == 1
        assert df.iloc[1]['loss'] == 0.4
    print(f'         write 2 rows, read back OK')


def test_config_loading():
    from src.utils.io import load_config

    cfg_path = 'configs/gmf.yaml'
    if os.path.exists(cfg_path):
        cfg = load_config(cfg_path)
        assert 'model' in cfg, f'{cfg_path}: missing "model" key'
        print(f'         {cfg_path}: model={cfg["model"]["name"]}')


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("HotelRec — Pipeline Validation (GMF)")
    print("=" * 60)
    print()

    print("--- Core imports ---")
    step("import torch, numpy, scipy, pandas, sklearn, yaml, tqdm", test_core_imports)

    print("\n--- GPU detection ---")
    step("CUDA availability", test_gpu_detection)

    print("\n--- Model creation ---")
    step("create GMF model", test_model_creation)

    print("\n--- GMF forward pass ---")
    step("GMF forward", test_gmf_forward)

    print("\n--- Ranking metrics ---")
    step("HR + NDCG", test_ranking_metrics)

    print("\n--- Checkpoint save/load ---")
    step("save and load GMF checkpoint", test_checkpoint_roundtrip)

    print("\n--- MetricsLogger ---")
    step("write + read CSV log", test_metrics_logger)

    print("\n--- Config loading ---")
    step("load YAML configs", test_config_loading)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("\nFix failures before submitting batch jobs!")
        sys.exit(1)
    else:
        print("\nAll pipeline checks passed! Ready to train:")
        print("  sbatch scripts/run_hpc.sh")
