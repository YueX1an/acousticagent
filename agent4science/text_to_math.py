"""Text-to-Math Bridge: translates LLM Critic JSON feedback into differentiable loss.

This module implements the formal mechanism described in Section 3.6 of the
research plan. It addresses three technical challenges:

1. Softplus activation (avoids gradient sparsity of max(0,·)²)
2. Dynamic τ scaling (prevents LLM from setting unrealistic targets)
3. Feedback smoothing via EMA (prevents loss landscape oscillation)

Core loss function:
    L(θ) = L_obj(θ)                                    # primary objective
         + Σ γ_j · Φ_j(f(θ))                           # hard constraint penalties
         + λ · Σ Softplus(τ_i - f_i(θ))               # LLM-guided penalties
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RefinementFeedback:
    """Structured feedback extracted from LLM Critic JSON output.

    This is the bridge data structure: LLM JSON → RefinementFeedback → loss terms.
    """
    verdict: str                                    # "pass" | "refine"
    physical_violations: list[dict] = field(default_factory=list)
    weak_regions: list[dict] = field(default_factory=list)
    causal_analysis: str = ""
    exploration_suggestion: str = ""
    confidence: float = 1.0
    raw_json: dict = field(default_factory=dict)

    @classmethod
    def from_llm_json(cls, data: dict) -> "RefinementFeedback":
        return cls(
            verdict=data.get("verdict", "refine"),
            physical_violations=data.get("physical_violations", []),
            weak_regions=data.get("weak_regions", []),
            causal_analysis=data.get("causal_analysis", ""),
            exploration_suggestion=data.get("exploration_suggestion", ""),
            confidence=data.get("confidence", 1.0),
            raw_json=data,
        )

    @property
    def is_pass(self) -> bool:
        return self.verdict == "pass"

    @property
    def has_actionable_feedback(self) -> bool:
        return len(self.weak_regions) > 0 or len(self.physical_violations) > 0


# ── Loss Function Components ──

def build_loss(
    param_tensor: torch.Tensor,
    model: torch.nn.Module,
    domain_spec,                      # DomainSpec
    feedback: RefinementFeedback,
    trust_factor: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Build the full loss function from LLM Critic feedback.

    Args:
        param_tensor: Current design parameters (1, input_dim), requires_grad=True.
        model: Surrogate model (frozen weights).
        domain_spec: DomainSpec for bounds and constraints.
        feedback: Structured feedback from LLM Critic.
        trust_factor: α ∈ (0, 1] — how aggressively to adopt LLM targets.

    Returns:
        (loss, loss_components) — loss is a scalar tensor for backprop,
        loss_components is a dict for logging.
    """
    # Forward pass
    output = model(param_tensor.unsqueeze(0) if param_tensor.dim() == 1 else param_tensor)
    output = output.flatten()

    # 1. Primary objective
    goal = domain_spec.optimization_goal
    if goal == "maximize_avg" or goal == "maximize_output":
        obj_score = torch.mean(output)
        loss_obj = -obj_score
    elif goal == "maximize_peak":
        obj_score = torch.max(output)
        loss_obj = -obj_score
    elif goal == "maximize_ratio":
        # For airfoil: maximize Cl/Cd
        if output.shape[0] >= 2:
            cl = output[0]
            cd = output[1] + 1e-6
            obj_score = cl / cd
            loss_obj = -obj_score
        else:
            loss_obj = -torch.mean(output)
            obj_score = torch.mean(output)
    else:
        loss_obj = -torch.mean(output)
        obj_score = torch.mean(output)

    # 2. Hard constraint penalties
    loss_constraints = torch.tensor(0.0, device=output.device)
    for j, violation in enumerate(feedback.physical_violations):
        gamma = float(violation.get("penalty_weight", 10.0))
        violation_type = violation.get("type", "unknown")
        loss_constraints += gamma * _compute_constraint_penalty(
            output, violation_type, domain_spec
        )

    # 3. LLM-guided Softplus penalties with dynamic τ scaling
    loss_refine = torch.tensor(0.0, device=output.device)
    n_weak = 0
    for region in feedback.weak_regions:
        indices = region.get("indices", [0, output.shape[0]])
        start, end = int(indices[0]), min(int(indices[1]), output.shape[0])
        if start >= end:
            continue

        region_output = output[start:end]
        current_mean = float(torch.mean(region_output).detach().cpu().item())

        # Dynamic τ scaling (Section 3.6.1)
        target_llm = float(region.get("target", 0.8))
        tau = min(1.0, current_mean + trust_factor * (target_llm - current_mean))

        # Softplus penalty (Section 3.6)
        weight = float(region.get("weight", 1.0))
        gap = tau - region_output
        softplus_penalty = F.softplus(gap).mean()  # Smooth, non-zero gradient everywhere

        loss_refine += weight * softplus_penalty
        n_weak += 1

    # Combined loss
    lambda_refine = 0.3 if n_weak > 0 else 0.0
    loss = loss_obj + 0.1 * loss_constraints + lambda_refine * loss_refine

    components = {
        "loss_obj": float(loss_obj.detach().cpu().item()),
        "loss_constraints": float(loss_constraints.detach().cpu().item()),
        "loss_refine": float(loss_refine.detach().cpu().item()),
        "n_weak_regions": n_weak,
        "obj_score": float(obj_score.detach().cpu().item()),
    }

    return loss, components


def _compute_constraint_penalty(
    output: torch.Tensor, violation_type: str, domain_spec
) -> torch.Tensor:
    """Compute penalty for a specific physical constraint violation.

    These are hard constraints with high fixed weights (γ_j).
    """
    if violation_type == "causality" or violation_type == "low_f_nonzero":
        # Low-frequency absorption should approach zero
        return torch.mean(output[:min(5, output.shape[0])]) ** 2

    elif violation_type == "absorption_bounded" or violation_type == "passivity":
        # Absorption should be in [0, 1]
        below = F.relu(-output)
        above = F.relu(output - 1.0)
        return (below ** 2).mean() + (above ** 2).mean()

    elif violation_type == "smoothness":
        # Penalize high-frequency oscillation
        if output.shape[0] >= 3:
            roughness = torch.mean(torch.abs(output[2:] - 2 * output[1:-1] + output[:-2]))
            return roughness
        return torch.tensor(0.0, device=output.device)

    elif violation_type == "Cl_negative" or violation_type == "Cd_negative":
        return F.relu(-output[0]).mean() + F.relu(-output[1]).mean()

    elif violation_type == "water_cement_ratio":
        return torch.tensor(0.0, device=output.device)  # Requires param access

    return torch.tensor(0.0, device=output.device)


# ── Feedback Smoothing (Section 3.6.2) ──

class FeedbackSmoother:
    """EMA-based smoothing of LLM Critic feedback across rounds.

    Prevents the loss landscape from undergoing step-function changes when
    the LLM outputs different weak_regions between rounds. Retains a decaying
    portion of previous-round penalties.
    """

    def __init__(self, decay: float = 0.5):
        self.decay = decay
        self._prev_weak_regions: list[dict] = []
        self._prev_verdict: str = "refine"

    def smooth(self, feedback: RefinementFeedback,
               current_output: np.ndarray) -> RefinementFeedback:
        """Smooth the current feedback with previous-round memory.

        For regions the LLM stopped flagging but haven't met targets,
        retain at decay*weight.
        """
        if not self._prev_weak_regions:
            # First round — no smoothing needed
            self._prev_weak_regions = feedback.weak_regions
            self._prev_verdict = feedback.verdict
            return feedback

        # Retain previous regions that still don't meet their targets
        smoothed_regions = list(feedback.weak_regions)  # Current regions at full weight

        current_region_indices = {
            (r.get("indices", [0, 0])[0], r.get("indices", [0, 0])[1])
            for r in feedback.weak_regions
        }

        for prev_region in self._prev_weak_regions:
            prev_start = prev_region.get("indices", [0, 0])[0]
            prev_end = prev_region.get("indices", [0, 0])[1]
            prev_key = (prev_start, prev_end)

            if prev_key in current_region_indices:
                continue  # Already in current regions

            # Check if previous region still needs help
            if prev_end <= len(current_output) and prev_start < prev_end:
                prev_current_mean = float(np.mean(current_output[prev_start:prev_end]))
                prev_target = prev_region.get("target", 0.8)

                if prev_current_mean < prev_target * 1.1:
                    # Retain at decayed weight
                    retained = dict(prev_region)
                    retained["weight"] = self.decay * float(prev_region.get("weight", 1.0))
                    retained["_retained"] = True
                    smoothed_regions.append(retained)

        # Update memory
        self._prev_weak_regions = smoothed_regions
        self._prev_verdict = feedback.verdict

        # Return smoothed feedback
        return RefinementFeedback(
            verdict=feedback.verdict,
            physical_violations=feedback.physical_violations,
            weak_regions=smoothed_regions,
            causal_analysis=feedback.causal_analysis,
            exploration_suggestion=feedback.exploration_suggestion,
            confidence=feedback.confidence,
            raw_json=feedback.raw_json,
        )

    def reset(self):
        self._prev_weak_regions = []
        self._prev_verdict = "refine"
