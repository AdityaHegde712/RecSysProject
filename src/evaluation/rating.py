"""
Rating prediction evaluation (RMSE, MAE) on the test set.

Complements the ranking metrics (HR@k, NDCG@k) with a traditional
explicit-feedback metric.

Supported models:
  * Any model with a `predict_rating(user_id, item_id)` method. ItemKNN and
    Popularity implement this naturally.
  * For ranking-only neural models (GMF, LightGCN, TextNCF),
    `calibrate_scores_to_ratings()` fits a linear regression score -> rating
    on the validation set, producing calibrated predictions so RMSE is
    computed on a consistent scale. The result is reported as a secondary
    number and labelled "calibrated" in outputs.
"""

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch


def _ratings_from_test(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["rating"].astype(np.float32).values


def rmse_from_predictions(preds: np.ndarray, truths: np.ndarray) -> float:
    diff = preds.astype(np.float64) - truths.astype(np.float64)
    return float(math.sqrt(np.mean(diff * diff)))


def mae_from_predictions(preds: np.ndarray, truths: np.ndarray) -> float:
    return float(np.mean(np.abs(preds.astype(np.float64) - truths.astype(np.float64))))


def evaluate_rating(
    model,
    test_df: pd.DataFrame,
    predict_fn: Optional[Callable] = None,
) -> dict[str, float]:
    """Evaluate RMSE and MAE on (user, item, rating) test tuples.

    If `predict_fn` is None, falls back to `model.predict_rating(u, i)`.
    Returns {'rmse': ..., 'mae': ..., 'n': ...}.
    """
    if predict_fn is None:
        if not hasattr(model, "predict_rating"):
            raise AttributeError(
                "Model has no `predict_rating` method. Pass predict_fn explicitly "
                "or use calibrate_scores_to_ratings() for ranking models."
            )
        predict_fn = model.predict_rating

    users = test_df["user_id"].values
    items = test_df["item_id"].values
    truths = _ratings_from_test(test_df)

    preds = np.empty(len(users), dtype=np.float32)
    for k in range(len(users)):
        preds[k] = float(predict_fn(int(users[k]), int(items[k])))

    return {
        "rmse": rmse_from_predictions(preds, truths),
        "mae": mae_from_predictions(preds, truths),
        "n": int(len(truths)),
    }


def _scores_for_pairs(
    model, users: np.ndarray, items: np.ndarray, device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Return model scores for arbitrary (user, item) pairs."""
    model_is_neural = hasattr(model, "parameters")
    if model_is_neural:
        model.eval()
        out = np.empty(len(users), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(users), batch_size):
                end = min(start + batch_size, len(users))
                u = torch.as_tensor(users[start:end], dtype=torch.long, device=device)
                i = torch.as_tensor(items[start:end], dtype=torch.long, device=device)
                out[start:end] = model(u, i).detach().cpu().numpy()
        return out
    # Non-neural: use predict_batch if available, else loop predict().
    if hasattr(model, "predict_batch"):
        u = torch.as_tensor(users, dtype=torch.long)
        i = torch.as_tensor(items, dtype=torch.long).unsqueeze(1)
        scores = model.predict_batch(u, i).numpy().ravel()
        return scores.astype(np.float32)
    return np.array([float(model.predict(int(u), int(i)))
                     for u, i in zip(users, items)], dtype=np.float32)


def calibrate_scores_to_ratings(
    model,
    val_df: pd.DataFrame,
    device: str = "cpu",
    rating_min: float = 1.0,
    rating_max: float = 5.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Fit a linear calibration score -> rating on validation pairs.

    Returns a predict_fn(users, items) -> predicted ratings (clipped to
    [rating_min, rating_max]).
    """
    vu = val_df["user_id"].values
    vi = val_df["item_id"].values
    vy = _ratings_from_test(val_df)

    vs = _scores_for_pairs(model, vu, vi, device=device)
    # Linear fit y = a*s + b.
    A = np.vstack([vs, np.ones_like(vs)]).T
    coef, *_ = np.linalg.lstsq(A, vy, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    def predict_fn(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        s = _scores_for_pairs(model, users, items, device=device)
        return np.clip(a * s + b, rating_min, rating_max)

    predict_fn.coef_a = a
    predict_fn.coef_b = b
    return predict_fn


def evaluate_rating_calibrated(
    model,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str = "cpu",
) -> dict[str, float]:
    """RMSE/MAE for ranking-only models via linear calibration on val split."""
    predict_fn = calibrate_scores_to_ratings(model, val_df, device=device)

    users = test_df["user_id"].values
    items = test_df["item_id"].values
    truths = _ratings_from_test(test_df)
    preds = predict_fn(users, items)

    return {
        "rmse_calibrated": rmse_from_predictions(preds, truths),
        "mae_calibrated": mae_from_predictions(preds, truths),
        "n": int(len(truths)),
        "calibration_a": float(predict_fn.coef_a),
        "calibration_b": float(predict_fn.coef_b),
    }
