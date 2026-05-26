"""Forward evaluation of trained surrogate model."""

import numpy as np
import torch

from ..models.resnet_model import ResNetDNN
from ..utils.metrics import compute_all_metrics


def evaluate_design(
    model: ResNetDNN,
    params: np.ndarray,
    device: torch.device | None = None,
) -> dict:
    """Evaluate absorption spectrum for a given design.

    Args:
        model: Pre-trained ResNetDNN surrogate model.
        params: 31-dim numpy array of design parameters.
        device: Torch device. Auto-detected if None.

    Returns:
        dict with keys: spectrum, metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.tensor(params, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        spectrum = model(tensor).flatten().cpu().numpy()

    spectrum = np.clip(spectrum, 0.0, 1.0)
    metrics = compute_all_metrics(spectrum)

    return {
        "spectrum": spectrum.tolist(),
        "metrics": metrics,
    }
