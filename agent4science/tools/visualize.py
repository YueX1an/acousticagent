"""Visualization utilities for absorption spectra and design results."""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..utils.physics import get_frequencies


def plot_spectrum(spectrum: np.ndarray, title: str = "Absorption Spectrum",
                  save_path: Optional[str] = None, show: bool = True,
                  comparison: Optional[dict] = None) -> None:
    """Plot a single absorption spectrum with optional comparison.

    Args:
        spectrum: 100-point absorption spectrum.
        title: Plot title.
        save_path: If provided, save figure to this path.
        show: Whether to call plt.show().
        comparison: Optional dict with 'label' and 'spectrum' for overlay.
    """
    freqs = get_frequencies()

    plt.figure(figsize=(12, 7))

    plt.plot(freqs, spectrum, 'r-', linewidth=3, label='Optimized Design')

    if comparison:
        plt.plot(freqs, comparison["spectrum"], 'b--', linewidth=2,
                 alpha=0.7, label=comparison.get("label", "Baseline"))

    # Mark peak
    peak_idx = np.argmax(spectrum)
    plt.plot(freqs[peak_idx], spectrum[peak_idx], 'ko', markersize=8)
    plt.text(freqs[peak_idx], spectrum[peak_idx] + 0.02,
             f'Peak: {spectrum[peak_idx]:.2f} @ {int(freqs[peak_idx])}Hz',
             ha='center', fontsize=11)

    # Mean line
    avg_val = np.mean(spectrum)
    plt.axhline(y=avg_val, color='blue', linestyle='--', alpha=0.5,
                label=f'Average: {avg_val:.3f}')

    # 0.8 threshold line
    plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.4,
                label='α = 0.8')

    plt.fill_between(freqs, spectrum, alpha=0.1, color='red')

    plt.title(title, fontsize=15)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Absorption Coefficient', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(0, 2000)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_agent_progress(round_log: list[dict],
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
    """Plot performance improvement over multi-agent debate rounds.

    Args:
        round_log: List of dicts with 'round', 'best_avg', 'best_peak'.
    """
    if not round_log:
        print("No round log to plot.")
        return

    rounds = [r["round"] for r in round_log]
    avgs = [r["best_avg"] for r in round_log]
    peaks = [r["best_peak"] for r in round_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds, avgs, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Debate Round', fontsize=12)
    ax1.set_ylabel('Average Absorption', fontsize=12)
    ax1.set_title('Average Absorption vs. Debate Rounds', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)

    ax2.plot(rounds, peaks, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Debate Round', fontsize=12)
    ax2.set_ylabel('Peak Absorption', fontsize=12)
    ax2.set_title('Peak Absorption vs. Debate Rounds', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_front(results: list[dict],
                       save_path: Optional[str] = None,
                       show: bool = True) -> None:
    """Plot Pareto front: avg_absorption vs peak_absorption across designs.

    Args:
        results: List of result dicts, each with 'summary' containing metrics.
    """
    avgs = [r["summary"]["avg_absorption"] for r in results]
    peaks = [r["summary"]["peak_absorption"] for r in results]
    labels = [r.get("label", f"Design {i+1}") for i, r in enumerate(results)]

    plt.figure(figsize=(8, 7))
    plt.scatter(avgs, peaks, c='blue', s=80, alpha=0.7)

    for i, label in enumerate(labels):
        plt.annotate(label, (avgs[i], peaks[i]),
                      textcoords="offset points", xytext=(5, 5), fontsize=9)

    plt.xlabel('Average Absorption', fontsize=12)
    plt.ylabel('Peak Absorption', fontsize=12)
    plt.title('Pareto Front: Broadband vs Peak Performance', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
