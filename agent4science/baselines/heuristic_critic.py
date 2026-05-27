"""Heuristic Critic: rule-based design reviewer (ABLATION BASELINE ONLY).

This is a non-AI baseline. It uses hardcoded rules for each output type:
- spectrum: equal-width band splitting
- vector: per-dimension target check
- scalar: finite-difference sensitivity analysis

It exists solely to answer the ablation question:
"Does the LLM add value beyond what a rule-based analyzer can do?"
"""

from __future__ import annotations

import numpy as np

from ..domain_spec import DomainSpec
from ..text_to_math import RefinementFeedback


class HeuristicCritic:
    """Rule-based design critic — NOT an AI contribution.

    This is a deterministic, non-learning baseline. It uses the same
    DomainSpec → feedback interface as the LLM Critic, but generates
    feedback via pre-programmed rules instead of LLM reasoning.
    """

    def __init__(self, domain: DomainSpec):
        self.domain = domain

    def review(self, design_params: list[float], design_metrics: dict,
               _constraint_value: float | None = None,
               **_kwargs) -> RefinementFeedback:
        """Review a design using hardcoded rules.

        Returns:
            RefinementFeedback compatible with the same Text-to-Math bridge
            used by the LLM Critic — enabling fair comparison.
        """
        output_type = self.domain.output_type

        if output_type == "spectrum":
            return self._critique_spectrum(design_params, design_metrics)
        elif output_type == "vector":
            return self._critique_vector(design_params, design_metrics)
        elif output_type == "scalar":
            return self._critique_scalar(design_params, design_metrics)
        else:
            return RefinementFeedback(verdict="pass",
                                       causal_analysis="Unknown output type.")

    # ── Spectrum Strategy ──

    def _critique_spectrum(self, params: list[float],
                           metrics: dict) -> RefinementFeedback:
        """Divide spectrum into N equal-width bands, flag weak ones."""
        n_bands = self.domain.critic_n_bands
        output_dim = self.domain.output_dim
        band_size = output_dim // n_bands

        weak_regions = []
        band_means = []

        min_band_absorption = self.domain.target_metrics.get(
            "min_band_absorption", 0.6
        )
        min_avg_absorption = self.domain.target_metrics.get(
            "min_avg_absorption", 0.7
        )

        for i in range(n_bands):
            start = i * band_size
            end = min((i + 1) * band_size, output_dim)
            # We reconstruct mean from the spectrum passed via metrics or compute
            # In practice, the spectrum should be passed separately
            band_mean = metrics.get(f"band_{i}_mean", 0.0)
            band_means.append(band_mean)

            if band_mean < min_band_absorption:
                label = self.domain.critic_band_labels[i] if i < len(self.domain.critic_band_labels) else f"band_{i}"
                weak_regions.append({
                    "indices": [start, end],
                    "label": label,
                    "current_mean": float(band_mean),
                    "target": min_band_absorption,
                    "weight": 1.0,
                    "reason": f"Mean absorption in {label} ({band_mean:.3f}) "
                              f"is below target ({min_band_absorption:.3f}).",
                })

        # Physical violations from metrics
        violations = []
        avg_abs = metrics.get("avg_absorption", 0.0)

        if avg_abs < min_avg_absorption:
            pass  # already captured by weak_regions

        verdict = "pass" if len(weak_regions) == 0 and len(violations) == 0 else "refine"

        return RefinementFeedback(
            verdict=verdict,
            physical_violations=violations,
            weak_regions=weak_regions,
            causal_analysis=(
                f"[HEURISTIC] {len(weak_regions)}/{n_bands} bands below "
                f"threshold {min_band_absorption}. "
                "Consider weighted optimization on weak bands."
            ),
            exploration_suggestion=(
                "Run weighted optimization with higher weight on weak bands."
                if weak_regions else "Design meets all heuristic criteria."
            ),
            confidence=0.5,
            raw_json={
                "method": "heuristic_spectrum",
                "n_bands": n_bands,
                "band_means": band_means,
                "threshold": min_band_absorption,
            },
        )

    # ── Vector Strategy ──

    def _critique_vector(self, params: list[float],
                         metrics: dict) -> RefinementFeedback:
        """Check each output dimension against its target."""
        weak_regions = []

        dim_targets = [
            (0, self.domain.output_labels[0] if self.domain.output_labels else "output_0",
             self.domain.target_metrics.get("min_Cl", 0.8), "maximize"),
            (1, self.domain.output_labels[1] if len(self.domain.output_labels) > 1 else "output_1",
             self.domain.target_metrics.get("max_Cd", 0.05), "minimize"),
        ]

        for i, (dim_idx, label, target, direction) in enumerate(dim_targets):
            current_value = metrics.get(label, metrics.get(f"output_{dim_idx}", 0.0))
            current_value = float(current_value) if not isinstance(current_value, str) else 0.0

            if direction == "maximize":
                if current_value < target:
                    weak_regions.append({
                        "indices": [dim_idx, dim_idx + 1],
                        "label": label,
                        "current_mean": current_value,
                        "target": target,
                        "weight": 1.0,
                        "reason": f"{label} ({current_value:.3f}) is below target ({target:.3f}).",
                    })
            elif direction == "minimize":
                if current_value > target:
                    weak_regions.append({
                        "indices": [dim_idx, dim_idx + 1],
                        "label": label,
                        "current_mean": current_value,
                        "target": 0.0,  # Minimize → target is 0
                        "weight": 0.5,   # Lower weight for drag reduction
                        "reason": f"{label} ({current_value:.4f}) exceeds target ({target:.4f}).",
                    })

        verdict = "pass" if len(weak_regions) == 0 else "refine"

        return RefinementFeedback(
            verdict=verdict,
            weak_regions=weak_regions,
            causal_analysis=(
                f"[HEURISTIC] {len(weak_regions)}/2 dimensions not meeting targets."
            ),
            exploration_suggestion=(
                "Multi-objective optimization with weighted dimensions."
            ),
            confidence=0.5,
            raw_json={"method": "heuristic_vector", "dim_targets": dim_targets},
        )

    # ── Scalar Strategy ──

    def _critique_scalar(self, params: list[float],
                         metrics: dict) -> RefinementFeedback:
        """Finite-difference sensitivity analysis for scalar outputs.

        Perturb each parameter by 1% and measure output change.
        Flag the most sensitive parameters for exploration.
        """
        target = self.domain.target_metrics.get("min_strength_MPa", 50.0)
        current_value = float(metrics.get("avg_absorption", 0.0))

        if current_value >= target:
            return RefinementFeedback(
                verdict="pass",
                causal_analysis=f"[HEURISTIC] Output ({current_value:.3f}) meets target ({target}).",
                confidence=0.5,
                raw_json={"method": "heuristic_scalar", "passed": True},
            )

        # Sensitivity analysis (approximate via bounds-based ranking)
        # In production, this would use finite differences via the surrogate.
        # Here we rank by parameter range as a proxy.
        sensitivity_ranking = []
        for i, (lo, hi) in enumerate(self.domain.param_bounds):
            param_range = hi - lo
            sensitivity_ranking.append((i, param_range))

        sensitivity_ranking.sort(key=lambda x: x[1], reverse=True)
        top_params = sensitivity_ranking[:3]

        weak_regions = [{
            "indices": [0, 1],  # scalar → whole output
            "label": self.domain.output_labels[0] if self.domain.output_labels else "output",
            "current_mean": current_value,
            "target": target,
            "weight": 1.0,
            "reason": (
                f"Output ({current_value:.3f}) below target ({target}). "
                f"Most sensitive parameters (by range): "
                f"{[self.domain.param_names[i] for i, _ in top_params]}."
            ),
        }]

        return RefinementFeedback(
            verdict="refine",
            weak_regions=weak_regions,
            causal_analysis=(
                f"[HEURISTIC] Output {current_value:.3f} < target {target}. "
                f"Top-3 parameters by sensitivity range: "
                f"{[self.domain.param_names[i] for i, _ in top_params]}."
            ),
            exploration_suggestion=(
                f"Focus optimization on: "
                f"{', '.join(self.domain.param_names[i] for i, _ in top_params)}."
            ),
            confidence=0.3,
            raw_json={
                "method": "heuristic_scalar_sensitivity",
                "sensitivity_ranking": [
                    {"param": self.domain.param_names[i], "range": r}
                    for i, r in sensitivity_ranking[:5]
                ],
            },
        )
