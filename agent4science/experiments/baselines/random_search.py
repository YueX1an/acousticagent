"""Random search baseline for inverse design.

Supports both legacy (hardcoded acoustic) and DomainSpec-based interfaces.
"""

from __future__ import annotations

import numpy as np
import torch

from ...models.resnet_model import ResNetDNN
from ...tools.evaluate import evaluate_design
from ...utils.physics import N_PARAMS, PARAM_BOUNDS, THICKNESS_INDEX
from ...domain_spec import DomainSpec


def random_search(model, domain=None, target_thickness_mm=None,
                  n_trials=5000, goal="maximize_avg",
                  device=None, seed=42) -> dict:
    """Random search baseline.

    Can be called with either:
    - DomainSpec: random_search(model, domain=domain_spec, ...)
    - Legacy acoustic: random_search(model, target_thickness_mm=100.0, ...)

    Args:
        model: Pre-trained ResNetDNN surrogate model.
        domain: DomainSpec (preferred).
        target_thickness_mm: Target thickness (legacy, acoustic-only).
        n_trials: Number of random designs.
        goal: Optimization goal.
        device: Torch device.
        seed: Random seed.

    Returns:
        dict with best design, metrics, and search statistics.
    """
    if domain is not None:
        return _random_search_domain(model, domain, n_trials, device, seed)
    return _random_search_legacy(model, target_thickness_mm, n_trials, goal, device, seed)


def _random_search_domain(
    model: ResNetDNN,
    domain: DomainSpec,
    n_trials: int = 5000,
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Random search using DomainSpec."""
    rng = np.random.RandomState(seed)
    bounds = domain.bounds_array
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    best_score = -np.inf
    best_params = None
    best_output = None
    all_scores = []

    for _ in range(n_trials):
        params = rng.uniform(lo, hi)
        for idx in domain.constrained_indices:
            params[idx] = lo[idx]  # Default to lower bound

        result = evaluate_design(model, params, device)
        output = np.array(result["spectrum"])

        if domain.optimization_goal in ("maximize_avg", "maximize_output"):
            score = float(np.mean(output))
        elif domain.optimization_goal == "maximize_peak":
            score = float(np.max(output))
        elif domain.optimization_goal == "maximize_ratio":
            score = float(output[0] / max(output[1], 1e-6)) if len(output) >= 2 else float(np.mean(output))
        else:
            score = float(np.mean(output))

        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_output = output.copy()

    best_result = evaluate_design(model, best_params, device)

    return {
        "best_params": best_params.tolist(),
        "best_spectrum": best_output.tolist(),
        "metrics": {"avg_output": float(np.mean(best_output)),
                     "peak_output": float(np.max(best_output))},
        "best_score": best_score,
        "search_stats": {
            "n_trials": n_trials,
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "max_score": float(np.max(all_scores)),
        },
    }


def _random_search_legacy(
    model: ResNetDNN,
    target_thickness_mm: float | None = None,
    n_trials: int = 5000,
    goal: str = "maximize_avg",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Legacy random search for acoustic domain (hardcoded 31-dim params)."""
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
