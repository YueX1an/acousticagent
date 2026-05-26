"""Physical constraints and equations for acoustic metamaterials."""

import numpy as np

# Speed of sound in air (m/s)
SPEED_OF_SOUND = 343.0

# Reference frequency range for spectrum (Hz)
FREQ_MIN = 0.0
FREQ_MAX = 2000.0
N_FREQ_POINTS = 100

# Parameter indices for the 31-dim design vector
# Indices 0-7:  Layer 1 cavity depths (8 params)
# Indices 8-15: Layer 2 cavity depths (8 params)
# Indices 16-19: Pore sizes group 1 (4 params)
# Indices 20-23: Pore sizes group 2 (4 params)
# Index 24: Total thickness (mm)
# Indices 25-26: Layer thickness ratios (2 params)
# Indices 27-30: Perforation ratios (4 params)
PARAM_LABELS = [
    "L1_Cav1", "L1_Cav2", "L1_Cav3", "L1_Cav4",
    "L1_Cav5", "L1_Cav6", "L1_Cav7", "L1_Cav8",
    "L2_Cav1", "L2_Cav2", "L2_Cav3", "L2_Cav4",
    "L2_Cav5", "L2_Cav6", "L2_Cav7", "L2_Cav8",
    "Pore1", "Pore2", "Pore3", "Pore4",
    "Pore5", "Pore6", "Pore7", "Pore8",
    "Thickness",
    "Ratio1", "Ratio2",
    "Perf1", "Perf2", "Perf3", "Perf4",
]

THICKNESS_INDEX = 24
N_PARAMS = 31

# Physical bounds for each parameter (min, max) in mm or ratio
PARAM_BOUNDS = np.array([
    [1.0, 100.0],   # Cavity depths (8x layer 1)
    [1.0, 100.0], [1.0, 100.0], [1.0, 100.0], [1.0, 100.0],
    [1.0, 100.0], [1.0, 100.0], [1.0, 100.0], [1.0, 100.0],
    [1.0, 100.0],   # Cavity depths (8x layer 2)
    [1.0, 100.0], [1.0, 100.0], [1.0, 100.0], [1.0, 100.0],
    [1.0, 100.0], [1.0, 100.0], [1.0, 100.0], [1.0, 100.0],
    [1.0, 80.0],    # Pore sizes
    [1.0, 80.0], [1.0, 80.0], [1.0, 80.0],
    [1.0, 80.0], [1.0, 80.0], [1.0, 80.0], [1.0, 80.0],
    [30.0, 300.0],  # Total thickness (mm)
    [1.0, 100.0],   # Layer ratios
    [1.0, 100.0],
    [1.0, 50.0],    # Perforation ratios
    [1.0, 50.0], [1.0, 50.0], [1.0, 50.0],
])


def clamp_params(params: np.ndarray, thickness_mm: float | None = None) -> np.ndarray:
    """Clamp design parameters to physical bounds and optionally lock thickness."""
    lower = PARAM_BOUNDS[:, 0]
    upper = PARAM_BOUNDS[:, 1]
    clamped = np.clip(params, lower, upper)
    if thickness_mm is not None:
        clamped[THICKNESS_INDEX] = thickness_mm
    return clamped


def validate_spectrum(spectrum: np.ndarray) -> dict:
    """Check if absorption spectrum satisfies physical constraints.

    Returns dict with keys: 'valid', 'issues', 'causality_score'.
    """
    issues = []

    # Absorption should be in [0, 1]
    if np.any(spectrum < -0.01):
        issues.append("Negative absorption detected")
    if np.any(spectrum > 1.01):
        issues.append("Absorption exceeds unity")

    # Spectrum should not oscillate wildly (smoothness check)
    roughness = np.mean(np.abs(np.diff(spectrum, n=2)))
    if roughness > 0.05:
        issues.append(f"High-frequency oscillation (roughness={roughness:.4f})")

    # Low-frequency limit: absorption → 0 as f → 0 (causality)
    low_f_absorption = np.mean(spectrum[:3])
    if low_f_absorption > 0.3:
        issues.append(f"Non-causal low-f limit: mean={low_f_absorption:.3f}")

    valid = len(issues) == 0
    return {
        "valid": valid,
        "issues": issues,
        "roughness": float(roughness),
        "low_f_absorption": float(low_f_absorption),
    }


def get_frequencies() -> np.ndarray:
    """Return the frequency grid (100 points, 0-2000 Hz)."""
    return np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ_POINTS)
