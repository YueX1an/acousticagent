"""DomainSpec: structured specification for arbitrary inverse design domains.

A DomainSpec captures everything the LLM Critic and Designer need to know
about a design problem — dimensionality, bounds, constraints, output topology,
and target metrics. From a DomainSpec, we auto-generate the structured prompt
sent to the LLM Critic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import yaml


OutputType = Literal["spectrum", "vector", "scalar"]


@dataclass
class DomainSpec:
    """Complete specification of an inverse design problem.

    One DomainSpec = one domain. To add a new domain, write a YAML file
    and load it — no code changes required.
    """

    # ── Identity ──
    name: str                              # "acoustic_metamaterial"
    description: str                       # Natural language domain description
    input_dim: int                         # Number of design parameters
    output_dim: int                        # Number of output dimensions

    # ── Parameter space ──
    param_bounds: list[list[float]]        # [[lo, hi], ...] — input_dim entries
    param_names: list[str]                 # Human-readable parameter labels
    constrained_indices: list[int] = field(default_factory=list)  # Locked params

    # ── Output structure ──
    output_type: OutputType = "spectrum"   # spectrum | vector | scalar
    output_x_axis: Optional[list[float]] = None   # For spectrum type (e.g. Hz)
    output_labels: list[str] = field(default_factory=list)  # Per-output-dim label

    # ── Physical constraints ──
    physical_constraints: list[str] = field(default_factory=list)

    # ── Optimization goal ──
    optimization_goal: str = "maximize_avg"  # maximize_avg | maximize_peak | maximize_output

    # ── Default initialization ──
    default_init_values: list[float] = field(default_factory=list)  # Per-param default

    # ── Critic configuration ──
    critic_n_bands: int = 5                # Number of output regions for spectrum
    critic_band_labels: list[str] = field(default_factory=list)

    # ── Target specifications ──
    target_metrics: dict[str, float] = field(default_factory=dict)
    init_strategies: list[str] = field(default_factory=lambda: ["default", "random"])

    # ── Surrogate config ──
    hidden_dim: int = 512
    num_blocks: int = 4

    def __post_init__(self):
        if not self.param_names:
            self.param_names = [f"param_{i}" for i in range(self.input_dim)]
        if not self.output_labels:
            self.output_labels = [f"output_{i}" for i in range(self.output_dim)]
        if not self.critic_band_labels and self.output_type == "spectrum":
            band_w = self.output_dim // self.critic_n_bands
            if self.output_x_axis and len(self.output_x_axis) >= self.output_dim:
                self.critic_band_labels = [
                    f"{self.output_x_axis[i * band_w]:.0f}-{self.output_x_axis[min((i+1)*band_w, self.output_dim-1)]:.0f}"
                    for i in range(self.critic_n_bands)
                ]
            else:
                self.critic_band_labels = [
                    f"band_{i}" for i in range(self.critic_n_bands)
                ]

    @property
    def bounds_array(self) -> np.ndarray:
        return np.array(self.param_bounds)

    # ── YAML I/O ──

    @classmethod
    def from_yaml(cls, path: str) -> "DomainSpec":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        data = {
            "name": self.name,
            "description": self.description,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "param_bounds": self.param_bounds,
            "param_names": self.param_names,
            "constrained_indices": self.constrained_indices,
            "output_type": self.output_type,
            "output_x_axis": self.output_x_axis,
            "output_labels": self.output_labels,
            "physical_constraints": self.physical_constraints,
            "optimization_goal": self.optimization_goal,
            "critic_n_bands": self.critic_n_bands,
            "critic_band_labels": self.critic_band_labels,
            "target_metrics": self.target_metrics,
            "init_strategies": self.init_strategies,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # ── Prompt generation ──

    def to_critic_prompt(self, current_params: list[float],
                         current_metrics: dict,
                         constraint_value: float | None = None) -> str:
        """Generate the structured prompt sent to the LLM Critic.

        This is the key method that makes DomainSpec an 'enabling interface.'
        The same prompt template works for ALL domains.
        """
        # Build parameter summary
        param_lines = []
        for i, (name, bounds) in enumerate(zip(self.param_names, self.param_bounds)):
            val = current_params[i] if i < len(current_params) else "?"
            locked = " [LOCKED]" if i in self.constrained_indices else ""
            param_lines.append(f"  {i:2d}. {name}: {val:.2f} (bounds: [{bounds[0]}, {bounds[1]}]){locked}")

        # Build output summary
        output_lines = []
        if self.output_type == "spectrum" and self.output_x_axis:
            output_lines.append(f"  Spectrum ({self.output_dim} points, {self.output_x_axis[0]:.0f}-{self.output_x_axis[-1]:.0f} {self._output_unit()})")
        elif self.output_type == "vector":
            for i, label in enumerate(self.output_labels):
                v = current_metrics.get(label, current_metrics.get(f"output_{i}", "?"))
                output_lines.append(f"  {label}: {v}")
        else:
            output_lines.append(f"  Output: {current_metrics}")

        # Build constraint info
        constraint_lines = []
        if self.constrained_indices:
            for idx in self.constrained_indices:
                constraint_lines.append(f"  Parameter {idx} ({self.param_names[idx]}): locked to {current_params[idx]:.2f}")
        if self.physical_constraints:
            for c in self.physical_constraints:
                constraint_lines.append(f"  {c}")

        prompt = f"""You are a scientific design reviewer specializing in {self.name}.

BACKGROUND:
{self.description}

DESIGN PROBLEM:
You are optimizing {self.input_dim} design parameters:
{chr(10).join(param_lines)}

The design produces {self.output_dim} output values.
{chr(10).join(output_lines)}

CONSTRAINTS:
{chr(10).join(constraint_lines) if constraint_lines else '  No constraints specified.'}

TARGET SPECIFICATIONS:
{self._format_targets()}

CURRENT DESIGN PERFORMANCE:
{self._format_metrics(current_metrics)}

YOUR TASK:
1. Is this design physically valid? Identify any violations.
2. Which aspects of performance are weakest relative to the targets?
3. Based on your physics knowledge, what parameter changes would most likely improve the weak aspects?
4. Verdict: "pass" (all targets met) or "refine" (improvement needed).

IMPORTANT: If your verdict is "refine", you MUST output a JSON block with the following structure:
```json
{{
  "verdict": "refine",
  "physical_violations": [
    {{"type": "violation_type", "description": "...", "penalty_weight": 10.0}}
  ],
  "weak_regions": [
    {{"indices": [start, end], "label": "region_name", "current_mean": 0.XX, "target": 0.XX, "weight": 1.5, "reason": "..."}}
  ],
  "causal_analysis": "Brief explanation of WHY the design is weak in these regions.",
  "exploration_suggestion": "Specific direction for improvement."
}}
```

If your verdict is "pass", output:
```json
{{"verdict": "pass", "summary": "All targets met."}}
```
"""
        return prompt

    def to_minimal_prompt(self, current_params: list[float],
                          current_metrics: dict) -> str:
        """Minimal prompt WITHOUT domain context — for ablation A3.
        Tests whether structured domain description matters.
        """
        return f"""You are a design reviewer.

Parameters ({self.input_dim}): {[f'{x:.2f}' for x in current_params[:8]]}...
Output ({self.output_dim}): {current_metrics}

Targets: {self._format_targets()}

Is this design acceptable? Output JSON with verdict (pass/refine) and weak_regions if refine.
"""

    # ── Helpers ──

    def _output_unit(self) -> str:
        domain_units = {
            "acoustic_metamaterial": "Hz",
            "airfoil_cst": "",
            "concrete_strength": "",
        }
        return domain_units.get(self.name, "")

    def _format_targets(self) -> str:
        if not self.target_metrics:
            return "  No specific targets."
        lines = []
        for k, v in self.target_metrics.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _format_metrics(self, metrics: dict) -> str:
        if not metrics:
            return "  No metrics available."
        lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── Factory function ──

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def load_domain(name: str) -> DomainSpec:
    """Load a domain specification by name.

    Args:
        name: One of 'acoustic', 'airfoil', 'concrete'.

    Returns:
        DomainSpec instance.
    """
    path = os.path.join(_CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Domain config not found: {path}. "
            f"Available: {[f.replace('.yaml','') for f in os.listdir(_CONFIG_DIR) if f.endswith('.yaml')]}"
        )
    return DomainSpec.from_yaml(path)
