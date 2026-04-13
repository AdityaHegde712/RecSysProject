# -*- coding: utf-8 -*-
"""Validate the HotelRec pipeline before submitting HPC jobs.

Tests the ItemKNN model: creation, fit on synthetic data, recommend,
predict, ranking metrics, and metrics logging.

Usage:
    python scripts/validate_pipeline.py

Run this after setup to catch issues before wasting compute hours.
"""

import sys
import os
import tempfile

import numpy as np
import pandas as pd

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
    import numpy
    import scipy
    import pandas
    import sklearn
    import yaml
    import tqdm


def test_model_creation():
    from src.models.common import build_model

    num_users, num_items = 100, 50
    cfg = {'model': {'name': 'itemknn', 'k_neighbors': 10}}
    model = build_model(cfg, num_users, num_items)
    assert model is not None, 'ItemKNN returned None'
    assert model.k == 10, f'expected k=10, got k={model.k}'
    print(f'         itemknn: k_neighbors={model.k}')


def test_itemknn_fit():
    from src.models.knn import ItemKNN

    num_users, num_items = 50, 30
    # create synthetic interactions
    rng = np.random.RandomState(42)
    n_interactions = 200
    users = rng.randint(0, num_users, n_interactions)
    items = rng.randint(0, num_items, n_interactions)
    df = pd.DataFrame({'user_id': users, 'item_id': items})

    model = ItemKNN(k_neighbors=10)
    model.fit(df, num_users, num_items)

    assert model.sim_matrix is not None, 'sim_matrix is None after fit'
    assert model.interaction_matrix is not None, 'interaction_matrix is None'
    assert model.sim_matrix.shape == (num_items, num_items)
    print(f'         sim_matrix: {model.sim_matrix.shape}, nnz={model.sim_matrix.nnz}')


def test_itemknn_recommend():
    from src.models.knn import ItemKNN

    num_users, num_items = 50, 30
    rng = np.random.RandomState(42)
    n_interactions = 200
    users = rng.randint(0, num_users, n_interactions)
    items = rng.randint(0, num_items, n_interactions)
    df = pd.DataFrame({'user_id': users, 'item_id': items})

    model = ItemKNN(k_neighbors=10)
    model.fit(df, num_users, num_items)

    recs = model.recommend(user_id=0, k=5)
    assert isinstance(recs, list), f'expected list, got {type(recs)}'
    assert len(recs) <= 5, f'expected <= 5 recs, got {len(recs)}'
    print(f'         recommend(user=0, k=5) -> {recs}')


def test_itemknn_predict():
    from src.models.knn import ItemKNN

    num_users, num_items = 50, 30
    rng = np.random.RandomState(42)
    n_interactions = 200
    users = rng.randint(0, num_users, n_interactions)
    items = rng.randint(0, num_items, n_interactions)
    df = pd.DataFrame({'user_id': users, 'item_id': items})

    model = ItemKNN(k_neighbors=10)
    model.fit(df, num_users, num_items)

    # scalar predict
    score = model.predict(0, 5)
    assert isinstance(score, float), f'expected float, got {type(score)}'

    # batch predict
    scores = model.predict([0, 1, 2], [5, 10, 15])
    assert len(scores) == 3, f'expected 3 scores, got {len(scores)}'
    print(f'         predict(scalar)={score:.4f}, predict(batch)={scores}')


def test_ranking_metrics():
    from src.evaluation.ranking import hit_ratio, ndcg

    ranked = [5, 3, 1, 7, 2]
    gt = 3

    hr5 = hit_ratio(ranked, gt, k=5)
    ndcg5 = ndcg(ranked, gt, k=5)
    assert hr5 == 1.0, f'HR@5 should be 1.0, got {hr5}'
    assert ndcg5 > 0, f'NDCG@5 should be > 0, got {ndcg5}'
    print(f'         HR@5={hr5:.2f}, NDCG@5={ndcg5:.4f}')


def test_model_save_load():
    from src.models.knn import ItemKNN
    from src.utils.io import save_model, load_model

    num_users, num_items = 50, 30
    rng = np.random.RandomState(42)
    users = rng.randint(0, num_users, 200)
    items = rng.randint(0, num_items, 200)
    df = pd.DataFrame({'user_id': users, 'item_id': items})

    model = ItemKNN(k_neighbors=10)
    model.fit(df, num_users, num_items)

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=True) as f:
        save_model(model, f.name)
        loaded = load_model(f.name)
        assert loaded.k == 10, f'expected k=10, got k={loaded.k}'
        assert loaded.num_users == num_users
        assert loaded.num_items == num_items

        # check that predictions match
        orig_score = model.predict(0, 5)
        loaded_score = loaded.predict(0, 5)
        assert abs(orig_score - loaded_score) < 1e-6
    print(f'         save + load: k={loaded.k}, predictions match')


def test_metrics_logger():
    from src.utils.metrics_logger import MetricsLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(tmpdir, filename='test_metrics.csv')
        logger.log(1, {'HR@10': 0.3, 'fit_time_s': 5.2})
        logger.log(2, {'HR@10': 0.4, 'fit_time_s': 5.1})

        df = logger.load()
        assert len(df) == 2, f'expected 2 rows, got {len(df)}'
        assert df.iloc[0]['epoch'] == 1
    print(f'         write 2 rows, read back OK')


def test_config_loading():
    from src.utils.io import load_config

    cfg_path = 'configs/itemknn.yaml'
    if os.path.exists(cfg_path):
        cfg = load_config(cfg_path)
        assert 'model' in cfg, f'{cfg_path}: missing "model" key'
        print(f'         {cfg_path}: model={cfg["model"]["name"]}')


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("HotelRec — Pipeline Validation (ItemKNN)")
    print("=" * 60)
    print()

    print("--- Core imports ---")
    step("import numpy, scipy, pandas, sklearn, yaml, tqdm", test_core_imports)

    print("\n--- Model creation ---")
    step("create ItemKNN model", test_model_creation)

    print("\n--- ItemKNN fit ---")
    step("fit on synthetic data", test_itemknn_fit)

    print("\n--- ItemKNN recommend ---")
    step("recommend top-k items", test_itemknn_recommend)

    print("\n--- ItemKNN predict ---")
    step("predict scores", test_itemknn_predict)

    print("\n--- Ranking metrics ---")
    step("HR + NDCG", test_ranking_metrics)

    print("\n--- Model save/load ---")
    step("save and load ItemKNN via pickle", test_model_save_load)

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
        print("  python -m src.train --config configs/itemknn.yaml --kcore 20")
