"""Bayesian Optimization baseline for inverse design.

Uses Gaussian Process with Matern kernel via scipy.optimize or skopt.
If skopt is unavailable, falls back to a simplified implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..domain_spec import DomainSpec
from ..tools.evaluate import evaluate_design


def bayesian_optimization(
    model: torch.nn.Module,
    domain: DomainSpec,
    n_iterations: int = 100,
    constraint_value: float | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    n_initial: int = 10,
) -> dict:
    """Bayesian Optimization with Gaussian Process surrogate.

    Uses Expected Improvement acquisition function.

    Args:
        model: Pre-trained surrogate model (forward predictor).
        domain: DomainSpec.
        n_iterations: Number of BO iterations.
        constraint_value: Locked parameter value.
        device: Torch device.
        seed: Random seed.
        n_initial: Number of random initial points.

    Returns:
        dict with best_params, metrics, optimization_history.
    """
    rng = np.random.RandomState(seed)
    bounds = domain.bounds_array
    input_dim = domain.input_dim

    # Normalize to [0, 1] for GP
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    def normalize(x):
        return (x - lo) / (hi - lo + 1e-10)

    def unnormalize(x_norm):
        return x_norm * (hi - lo) + lo

    # Initial random samples
    X_norm = rng.uniform(0, 1, (n_initial, input_dim))
    X = np.array([unnormalize(x) for x in X_norm])

    if constraint_value is not None:
        for idx in domain.constrained_indices:
            X[:, idx] = constraint_value

    X = np.clip(X, lo, hi)

    # Evaluate initial samples
    y = np.zeros(n_initial)
    for i in range(n_initial):
        result = evaluate_design(model, X[i], device)
        y[i] = _compute_objective(result["spectrum"], domain)

    # Best so far
    best_idx = np.argmax(y)
    best_x = X[best_idx].copy()
    best_y = y[best_idx]

    history = [{"iteration": 0, "best_y": float(best_y)}]

    # BO loop
    for iteration in range(1, n_iterations + 1):
        try:
            # Fit GP
            gp = _fit_gp(X_norm, y, rng)

            # Find next point via Expected Improvement
            x_next_norm = _maximize_ei(gp, X_norm, y, bounds_normalized=True, rng=rng)
            x_next = unnormalize(x_next_norm)

            if constraint_value is not None:
                for idx in domain.constrained_indices:
                    x_next[idx] = constraint_value
            x_next = np.clip(x_next, lo, hi)

            # Evaluate
            result = evaluate_design(model, x_next, device)
            y_next = _compute_objective(result["spectrum"], domain)

            # Update
            X_norm = np.vstack([X_norm, normalize(x_next)])
            y = np.append(y, y_next)

            if y_next > best_y:
                best_y = y_next
                best_x = x_next.copy()

            history.append({"iteration": iteration, "best_y": float(best_y)})

        except Exception:
            # Fallback: random sample
            x_next_norm = rng.uniform(0, 1, input_dim)
            x_next = unnormalize(x_next_norm)
            if constraint_value is not None:
                for idx in domain.constrained_indices:
                    x_next[idx] = constraint_value
            x_next = np.clip(x_next, lo, hi)
            result = evaluate_design(model, x_next, device)
            y_next = _compute_objective(result["spectrum"], domain)
            X_norm = np.vstack([X_norm, normalize(x_next)])
            y = np.append(y, y_next)

    # Final evaluation
    final_result = evaluate_design(model, best_x, device)

    return {
        "best_params": best_x.tolist(),
        "best_spectrum": final_result["spectrum"],
        "metrics": final_result["metrics"],
        "best_score": float(best_y),
        "optimization_history": history,
    }


def _compute_objective(spectrum: list, domain: DomainSpec) -> float:
    """Compute domain-specific objective from spectrum."""
    s = np.array(spectrum, dtype=np.float64)
    goal = domain.optimization_goal

    if goal == "maximize_avg" or goal == "maximize_output":
        return float(np.mean(s))
    elif goal == "maximize_peak":
        return float(np.max(s))
    elif goal == "maximize_ratio":
        if len(s) >= 2:
            return float(s[0] / max(s[1], 1e-6))
        return float(np.mean(s))
    return float(np.mean(s))


# ── Simplified GP components (no external dependency) ──

def _fit_gp(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState) -> dict:
    """Fit a simplified Gaussian Process.

    Uses a Matern-like RBF kernel with length scale estimated from data.
    Returns a dict with kernel parameters (not a full GP object).
    """
    n = X.shape[0]
    d = X.shape[1]

    # Estimate length scale
    if n > 1:
        dists = []
        for i in range(min(n, 50)):
            for j in range(i + 1, min(n, 50)):
                dists.append(np.sum((X[i] - X[j]) ** 2))
        length_scale = np.sqrt(np.median(dists)) if dists else 1.0
    else:
        length_scale = 1.0

    length_scale = max(length_scale, 0.01)

    # Signal variance
    signal_var = np.var(y) if n > 1 else 1.0
    signal_var = max(signal_var, 0.01)

    # Noise variance
    noise_var = 0.01 * signal_var

    # Precompute kernel matrix
    K = _rbf_kernel(X, X, length_scale, signal_var)
    K += noise_var * np.eye(n)

    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        K += 0.1 * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    return {
        "length_scale": length_scale,
        "signal_var": signal_var,
        "noise_var": noise_var,
        "alpha": alpha,
        "L": L,
        "X_train": X,
        "y_train": y,
    }


def _rbf_kernel(X1: np.ndarray, X2: np.ndarray,
                length_scale: float, signal_var: float) -> np.ndarray:
    """RBF (Matern-like) kernel."""
    dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1).reshape(1, -1) - \
            2 * np.dot(X1, X2.T)
    return signal_var * np.exp(-0.5 * dists / (length_scale ** 2 + 1e-10))


def _gp_predict(gp: dict, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GP posterior mean and variance."""
    k_star = _rbf_kernel(X_new, gp["X_train"],
                         gp["length_scale"], gp["signal_var"])
    k_star_star = _rbf_kernel(X_new, X_new,
                              gp["length_scale"], gp["signal_var"])

    mean = k_star @ gp["alpha"]
    v = np.linalg.solve(gp["L"], k_star.T)
    var = np.diag(k_star_star) - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 1e-10)

    return mean, var


def _maximize_ei(gp: dict, X: np.ndarray, y: np.ndarray,
                 bounds_normalized: bool = True,
                 n_restarts: int = 10,
                 n_candidates: int = 1000,
                 rng: np.random.RandomState | None = None) -> np.ndarray:
    """Maximize Expected Improvement via random search + local refinement."""
    if rng is None:
        rng = np.random.RandomState()

    d = X.shape[1]
    y_best = np.max(y)
    mean_all, var_all = _gp_predict(gp, X)
    xi = 0.01  # Exploration parameter

    best_ei = -np.inf
    best_x = None

    for _ in range(n_restarts):
        # Random candidate
        if best_x is not None:
            # Local search around current best
            x_candidates = best_x + rng.normal(0, 0.1, (n_candidates, d))
            x_candidates = np.clip(x_candidates, 0, 1)
        else:
            x_candidates = rng.uniform(0, 1, (n_candidates, d))

        mu, var = _gp_predict(gp, x_candidates)
        sigma = np.sqrt(var)

        # Expected Improvement
        improvement = mu - y_best - xi
        Z = np.divide(improvement, sigma, out=np.zeros_like(sigma), where=sigma > 1e-10)
        ei = improvement * _norm_cdf(Z) + sigma * _norm_pdf(Z)

        idx = np.argmax(ei)
        if ei[idx] > best_ei:
            best_ei = ei[idx]
            best_x = x_candidates[idx].copy()

    if best_x is None:
        best_x = rng.uniform(0, 1, d)

    return best_x


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
