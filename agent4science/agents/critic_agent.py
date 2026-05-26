"""Critic Agent: validates designs and identifies weaknesses for refinement."""

import numpy as np

from .base_agent import BaseAgent, AgentMessage, AgentConfig


class CriticAgent(BaseAgent):
    """The Critic Agent analyzes absorption spectra and identifies weaknesses.

    It checks:
    1. Physical validity (causality, passivity)
    2. Spectral weaknesses (bands with low absorption)
    3. Comparison against design targets
    4. Convergence assessment (is further refinement needed?)
    """

    def __init__(self):
        config = AgentConfig(
            name="Critic",
            role="Design Validator & Critique Provider",
            system_prompt=(
                "You are a rigorous design reviewer. "
                "You validate absorption spectra against physical constraints "
                "and identify specific weaknesses for improvement."
            ),
        )
        super().__init__(config)

    def act(self, input_message: AgentMessage) -> AgentMessage:
        """Analyze a design and provide critique.

        Expected input_message.content:
            - design: dict with 'spectrum', 'metrics', 'optimized_params'
            - target_spec: dict with target performance requirements
            - round_id: int

        Returns:
            AgentMessage with critique: pass/fail, weak_bands, suggestions
        """
        from ..tools.validate import validate_design
        from ..utils.physics import N_FREQ_POINTS

        design = input_message.content.get("design", {})
        target_spec = input_message.content.get("target_spec", {})
        spectrum = np.array(design.get("spectrum", []))
        params = np.array(design.get("optimized_params", []))
        thickness = design.get("thickness_mm", 100.0)

        if len(spectrum) == 0:
            return AgentMessage(
                sender=self.name,
                recipient=input_message.sender,
                content={"verdict": "fail", "reason": "No spectrum provided"},
                round_id=input_message.round_id,
            )

        # 1. Physical validation
        if len(params) > 0:
            validation = validate_design(params, spectrum, thickness)
        else:
            validation = {"valid": True, "issues": [], "warnings": []}

        # 2. Identify weak bands (divide spectrum into 5 bands)
        n_bands = 5
        band_size = N_FREQ_POINTS // n_bands
        band_means = []
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            band_means.append(float(np.mean(spectrum[start:end])))

        weak_bands = [
            i for i, m in enumerate(band_means)
            if m < target_spec.get("min_band_absorption", 0.7)
        ]

        # 3. Check overall metrics against targets
        metrics = design.get("metrics", {})
        avg_abs = metrics.get("avg_absorption", 0.0)
        peak_abs = metrics.get("peak_absorption", 0.0)
        bw_08 = metrics.get("bandwidth_0.8_hz", 0.0)

        targets_met = True
        failures = []

        if avg_abs < target_spec.get("min_avg_absorption", 0.7):
            targets_met = False
            failures.append(f"Avg absorption {avg_abs:.3f} < target {target_spec.get('min_avg_absorption', 0.7)}")

        if peak_abs < target_spec.get("min_peak_absorption", 0.9):
            targets_met = False
            failures.append(f"Peak absorption {peak_abs:.3f} < target {target_spec.get('min_peak_absorption', 0.9)}")

        # 4. Determine if refinement is needed
        needs_refinement = not targets_met or len(weak_bands) > 0 or not validation["valid"]

        verdict = "pass" if not needs_refinement else "refine"

        # 5. Generate actionable suggestions
        suggestions = self._generate_suggestions(
            spectrum, band_means, weak_bands, failures, validation
        )

        return AgentMessage(
            sender=self.name,
            recipient=input_message.sender,
            content={
                "verdict": verdict,
                "targets_met": targets_met,
                "weak_bands": weak_bands,
                "band_means": band_means,
                "failures": failures,
                "validation": validation,
                "suggestions": suggestions,
                "metrics_summary": {
                    "avg_absorption": avg_abs,
                    "peak_absorption": peak_abs,
                    "bandwidth_0.8": bw_08,
                    "worst_band": min(band_means),
                },
            },
            round_id=input_message.round_id,
        )

    def _generate_suggestions(self, spectrum: np.ndarray,
                               band_means: list[float],
                               weak_bands: list[int],
                               failures: list[str],
                               validation: dict) -> list[str]:
        """Generate specific improvement suggestions."""
        suggestions = []

        freq_step = 20.0
        band_freq_ranges = {
            0: "0-400 Hz",
            1: "400-800 Hz",
            2: "800-1200 Hz",
            3: "1200-1600 Hz",
            4: "1600-2000 Hz",
        }

        for band_idx in weak_bands:
            freq_range = band_freq_ranges.get(band_idx, f"band {band_idx}")
            current = band_means[band_idx]
            suggestions.append(
                f"Weak absorption in {freq_range}: current avg={current:.3f}. "
                "Consider adjusting cavity depths to target this frequency range."
            )

        # Physical validity suggestions
        if not validation.get("valid", True):
            for issue in validation.get("issues", []):
                suggestions.append(f"Physical issue: {issue}")

        # Peak-specific suggestions
        peak_idx = int(np.argmax(spectrum))
        peak_hz = peak_idx * freq_step
        if peak_hz < 200:
            suggestions.append(
                f"Peak at very low frequency ({peak_hz} Hz). "
                "Consider reducing cavity depths to shift resonance higher."
            )

        return suggestions
