"""Evaluation metrics for absorption spectra."""

import numpy as np


def avg_absorption(spectrum: np.ndarray) -> float:
    """Mean absorption coefficient across all frequencies."""
    return float(np.mean(spectrum))


def peak_absorption(spectrum: np.ndarray) -> float:
    """Maximum absorption coefficient."""
    return float(np.max(spectrum))


def peak_frequency(spectrum: np.ndarray, freq_step: float = 20.0) -> int:
    """Frequency (Hz) at which peak absorption occurs."""
    return int(np.argmax(spectrum) * freq_step)


def bandwidth_above(spectrum: np.ndarray, threshold: float = 0.8,
                    freq_step: float = 20.0) -> float:
    """Bandwidth (Hz) where absorption exceeds threshold."""
    above = spectrum >= threshold
    if not np.any(above):
        return 0.0
    return float(np.sum(above) * freq_step)


def worst_band_absorption(spectrum: np.ndarray, n_bands: int = 5) -> float:
    """Minimum average absorption across n_bands equal frequency bands.

    This penalizes designs with narrow-band performance.
    """
    band_size = len(spectrum) // n_bands
    band_means = [np.mean(spectrum[i * band_size:(i + 1) * band_size])
                  for i in range(n_bands)]
    return float(np.min(band_means))


def spectral_smoothness(spectrum: np.ndarray) -> float:
    """Measure spectral smoothness (lower = smoother)."""
    return float(np.mean(np.abs(np.diff(spectrum, n=2))))


def compute_all_metrics(spectrum: np.ndarray, freq_step: float = 20.0) -> dict:
    """Compute all evaluation metrics for a spectrum."""
    avg = avg_absorption(spectrum)
    peak = peak_absorption(spectrum)
    return {
        "avg_absorption": avg,
        "avg_output": avg,      # Alias for cross-domain compatibility
        "peak_absorption": peak,
        "peak_output": peak,    # Alias for cross-domain compatibility
        "peak_frequency_hz": peak_frequency(spectrum, freq_step),
        "bandwidth_0.8_hz": bandwidth_above(spectrum, 0.8, freq_step),
        "bandwidth_0.9_hz": bandwidth_above(spectrum, 0.9, freq_step),
        "worst_band_absorption": worst_band_absorption(spectrum),
        "spectral_smoothness": spectral_smoothness(spectrum),
    }
