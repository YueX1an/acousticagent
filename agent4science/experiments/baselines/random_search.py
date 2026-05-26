"""Random search baseline for inverse design."""

import numpy as np
import torch

from ...models.resnet_model import ResNetDNN
from ...tools.evaluate import evaluate_design
from ...utils.physics import N_PARAMS, PARAM_BOUNDS, THICKNESS_INDEX


def random_search(
    model: ResNetDNN,
    target_thickness_mm: float,
    n_trials: int = 5000,
    goal: str = "maximize_avg",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Simple random search baseline.

    Samples random designs uniformly within parameter bounds,
    evaluates each via the surrogate model, and returns the best.

    Args:
        model: Pre-trained ResNetDNN surrogate model.
        target_thickness_mm: Target thickness constraint.
        n_trials: Number of random designs to evaluate.
        goal: Optimization goal ('maximize_avg' or 'maximize_peak').
        device: Torch device.
        seed: Random seed for reproducibility.

    Returns:
        dict with best design, metrics, and search statistics.
    """
    rng = np.random.RandomState(seed)

    best_score = -np.inf
    best_params = None
    best_spectrum = None
    all_scores = []

    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]

    for i in range(n_trials):
        params = rng.uniform(lo, hi)
        params[THICKNESS_INDEX] = target_thickness_mm

        result = evaluate_design(model, params, device)
        spectrum = np.array(result["spectrum"])

        if goal == "maximize_avg":
            score = float(np.mean(spectrum))
        elif goal == "maximize_peak":
            score = float(np.max(spectrum))
        else:
            score = float(np.mean(spectrum))

        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_spectrum = spectrum.copy()

    best_result = evaluate_design(model, best_params, device)

    return {
        "best_params": best_params.tolist(),
        "best_spectrum": best_spectrum.tolist(),
        "metrics": best_result["metrics"],
        "best_score": best_score,
        "search_stats": {
            "n_trials": n_trials,
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "max_score": float(np.max(all_scores)),
        },
    }
