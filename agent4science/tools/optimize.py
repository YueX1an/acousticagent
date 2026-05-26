"""Gradient-based inverse design optimization."""

import numpy as np
import torch
import torch.optim as optim

from ..models.resnet_model import ResNetDNN
from ..utils.physics import N_PARAMS, PARAM_BOUNDS, THICKNESS_INDEX, clamp_params
from ..utils.metrics import compute_all_metrics


def optimize_design(
    model: ResNetDNN,
    target_thickness_mm: float,
    goal: str = "maximize_avg",
    init_params: np.ndarray | None = None,
    steps: int = 100,
    lr: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Gradient-based inverse design optimization.

    Args:
        model: Pre-trained ResNetDNN surrogate model (frozen weights).
        target_thickness_mm: Constraint on total thickness.
        goal: 'maximize_avg' for broadband or 'maximize_peak' for resonance.
        init_params: Initial guess (31-dim numpy array). If None, uses defaults.
        steps: Number of Adam optimization steps.
        lr: Learning rate.
        device: Torch device. Auto-detected if None.
        verbose: Print progress messages.

    Returns:
        dict with keys: optimized_params, spectrum, metrics, best_score, convergence.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default initialization
    if init_params is None:
        init_params = np.full(N_PARAMS, 10.0)
        init_params[24] = target_thickness_mm

    init_params = clamp_params(init_params.copy(), target_thickness_mm)

    param_tensor = torch.tensor(
        init_params, dtype=torch.float32, device=device, requires_grad=True
    )

    optimizer = optim.Adam([param_tensor], lr=lr)

    best_score = -np.inf
    best_spectrum = None
    best_params = None

    for i in range(steps):
        optimizer.zero_grad()

        spectrum = model(param_tensor.unsqueeze(0)).flatten()

        if goal == "maximize_avg":
            score = torch.mean(spectrum)
        elif goal == "maximize_peak":
            score = torch.max(spectrum)
        elif goal == "maximize_band":
            # Maximize fraction of spectrum above 0.8
            score = torch.mean((spectrum > 0.8).float())
        else:
            raise ValueError(f"Unknown goal: {goal}")

        loss = -score
        loss.backward()
        optimizer.step()

        # Project onto physical constraints
        with torch.no_grad():
            param_tensor.data[THICKNESS_INDEX] = target_thickness_mm
            lower = torch.tensor(PARAM_BOUNDS[:, 0], dtype=torch.float32, device=device)
            upper = torch.tensor(PARAM_BOUNDS[:, 1], dtype=torch.float32, device=device)
            param_tensor.data = torch.clamp(param_tensor.data, lower, upper)

        if score.item() > best_score:
            best_score = score.item()
            best_spectrum = spectrum.detach().cpu().numpy()
            best_params = param_tensor.detach().cpu().numpy().copy()

        if verbose and i % 20 == 0:
            print(f"   Step {i:3d}/{steps}: Score = {score.item():.4f}")

    best_spectrum = np.clip(best_spectrum, 0.0, 1.0)
    metrics = compute_all_metrics(best_spectrum)

    return {
        "optimized_params": best_params.tolist(),
        "spectrum": best_spectrum.tolist(),
        "metrics": metrics,
        "best_score": float(best_score),
        "goal": goal,
        "thickness_mm": target_thickness_mm,
    }
