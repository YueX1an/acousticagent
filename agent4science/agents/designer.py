"""Designer: domain-agnostic gradient-based optimization module.

This is an ALGORITHMIC module — not an AI agent. It performs multi-start
gradient descent using Adam, reading DomainSpec for problem dimensions,
parameter bounds, constraint indices, and optimization goal.

Supports LLM-guided refinement via the Text-to-Math Bridge (build_loss).
When no LLM feedback is available, falls back to pure objective optimization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from ..domain_spec import DomainSpec
from ..text_to_math import RefinementFeedback, FeedbackSmoother, build_loss
from ..tools.evaluate import evaluate_design


class Designer:
    """Domain-agnostic gradient-based inverse design optimizer.

    Key design: reads DomainSpec for ALL problem-specific information.
    No hardcoded dimensions, bounds, or constraint indices.
    """

    def __init__(self, model: torch.nn.Module, domain: DomainSpec,
                 device: torch.device | None = None,
                 n_restarts: int = 5, steps_per_restart: int = 100,
                 lr: float = 0.5):
        """
        Args:
            model: Pre-trained surrogate model (frozen weights).
            domain: DomainSpec describing the design problem.
            device: Torch device.
            n_restarts: Number of random restarts per optimization.
            steps_per_restart: Adam steps per restart.
            lr: Learning rate.
        """
        self.model = model
        self.domain = domain
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_restarts = n_restarts
        self.steps_per_restart = steps_per_restart
        self.lr = lr

        self._smoother = FeedbackSmoother(decay=0.5)

    def optimize(self, feedback: RefinementFeedback | None = None,
                 constraint_value: float | None = None) -> dict:
        """Run multi-start gradient descent optimization.

        Args:
            feedback: Optional LLM Critic feedback for weighted refinement.
                If None, runs pure objective optimization (Designer-Only mode).
            constraint_value: Value to lock constrained params to.

        Returns:
            dict with: optimized_params, spectrum, metrics, best_score.
        """
        bounds = self.domain.bounds_array
        constrained = self.domain.constrained_indices

        best_result = None
        best_score = -np.inf

        # Apply feedback smoothing if we have feedback
        if feedback:
            # We need the current best output for the smoother
            feedback = self._smoother.smooth(
                feedback, np.zeros(self.domain.output_dim)
            )

        for restart in range(self.n_restarts):
            # Generate initialization
            strategy = self.domain.init_strategies[
                restart % len(self.domain.init_strategies)
            ]
            init_params = self._generate_init(strategy, restart, constraint_value)

            # Optimize
            if feedback and feedback.has_actionable_feedback:
                result = self._optimize_with_feedback(
                    init_params, feedback, constraint_value, restart
                )
            else:
                result = self._optimize_pure(
                    init_params, constraint_value, restart
                )

            score = self._extract_score(result)
            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise RuntimeError("Optimization produced no results.")

        best_result["n_restarts"] = self.n_restarts
        return best_result

    # ── Optimization Kernels ──

    def _optimize_pure(self, init_params: np.ndarray,
                       constraint_value: float | None,
                       restart_idx: int) -> dict:
        """Pure objective optimization (no LLM feedback)."""
        param_tensor = torch.tensor(
            init_params, dtype=torch.float32, device=self.device, requires_grad=True
        )
        opt = optim.Adam([param_tensor], lr=self.lr)

        best_score = -np.inf
        best_params = None
        best_output = None

        for step in range(self.steps_per_restart):
            opt.zero_grad()
            output = self._forward(param_tensor)
            loss, score = self._objective_loss(output)

            loss.backward()
            opt.step()

            self._project(param_tensor, constraint_value)
            score_val = score.item() if isinstance(score, torch.Tensor) else float(score)

            if score_val > best_score:
                best_score = score_val
                best_output = output.detach().cpu().numpy().copy()
                best_params = param_tensor.detach().cpu().numpy().copy()

        return self._build_result(best_params, best_output, best_score)

    def _optimize_with_feedback(self, init_params: np.ndarray,
                                feedback: RefinementFeedback,
                                constraint_value: float | None,
                                restart_idx: int) -> dict:
        """Weighted optimization using LLM Critic feedback.

        Uses the Text-to-Math Bridge to construct a loss function from
        structured LLM feedback.
        """
        param_tensor = torch.tensor(
            init_params, dtype=torch.float32, device=self.device, requires_grad=True
        )
        opt = optim.Adam([param_tensor], lr=self.lr)

        best_score = -np.inf
        best_params = None
        best_output = None
        best_overall = -np.inf

        for step in range(self.steps_per_restart):
            opt.zero_grad()

            # Use Text-to-Math Bridge for combined loss
            trust_factor = 0.3 + 0.1 * (restart_idx)  # Increase trust with restarts
            loss, components = build_loss(
                param_tensor, self.model, self.domain, feedback, trust_factor
            )

            loss.backward()
            opt.step()

            self._project(param_tensor, constraint_value)

            # Track best by overall score (primary objective, not combined loss)
            with torch.no_grad():
                output = self._forward(param_tensor)
                _, obj_score = self._objective_loss(output)
                obj_val = obj_score.item()

            if obj_val > best_score:
                best_score = obj_val
                best_output = output.detach().cpu().numpy().copy()
                best_params = param_tensor.detach().cpu().numpy().copy()

            overall = -loss.item()
            if overall > best_overall:
                best_overall = overall

        return self._build_result(best_params, best_output, best_score)

    # ── Helpers ──

    def _forward(self, param_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through surrogate model."""
        x = param_tensor.unsqueeze(0) if param_tensor.dim() == 1 else param_tensor
        return self.model(x).flatten()

    def _objective_loss(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute primary objective loss based on DomainSpec.optimization_goal.

        Returns:
            (loss, score) — loss is minimized, score is tracked for best selection.
        """
        goal = self.domain.optimization_goal

        if goal == "maximize_avg" or goal == "maximize_output":
            score = torch.mean(output)
            loss = -score
        elif goal == "maximize_peak":
            score = torch.max(output)
            loss = -score
        elif goal == "maximize_ratio":
            if output.shape[0] >= 2:
                cl = output[0]
                cd = output[1] + 1e-6
                score = cl / cd
                loss = -score
            else:
                score = torch.mean(output)
                loss = -score
        else:
            score = torch.mean(output)
            loss = -score

        return loss, score

    def _project(self, param_tensor: torch.Tensor,
                 constraint_value: float | None):
        """Project parameters onto physical bounds and lock constraints."""
        with torch.no_grad():
            lo = torch.tensor(self.domain.bounds_array[:, 0],
                              dtype=torch.float32, device=self.device)
            hi = torch.tensor(self.domain.bounds_array[:, 1],
                              dtype=torch.float32, device=self.device)
            param_tensor.data = torch.clamp(param_tensor.data, lo, hi)

            if constraint_value is not None:
                for idx in self.domain.constrained_indices:
                    param_tensor.data[idx] = constraint_value

    def _generate_init(self, strategy: str, restart_idx: int,
                       constraint_value: float | None) -> np.ndarray:
        """Generate initial parameters."""
        lo = self.domain.bounds_array[:, 0]
        hi = self.domain.bounds_array[:, 1]
        input_dim = self.domain.input_dim

        if strategy == "random":
            params = np.random.uniform(lo, hi)
        elif strategy == "theory_guided":
            # Use default init values if available, otherwise mid-range
            if self.domain.default_init_values:
                base = np.array(self.domain.default_init_values)
            else:
                base = (lo + hi) / 2.0
            params = base + np.random.uniform(-0.1, 0.1, input_dim) * (hi - lo)
        else:  # default
            if self.domain.default_init_values:
                base = np.array(self.domain.default_init_values)
            else:
                base = (lo + hi) / 2.0
            if restart_idx > 0:
                params = base + np.random.uniform(-0.05, 0.05, input_dim) * (hi - lo)
            else:
                params = base.copy()

        # Apply constraint
        if constraint_value is not None:
            for idx in self.domain.constrained_indices:
                params[idx] = constraint_value

        # Clamp to bounds
        params = np.clip(params, lo, hi)
        return params

    def _extract_score(self, result: dict) -> float:
        """Extract the primary score from a result dict for comparison."""
        return float(result.get("best_score", -np.inf))

    def _build_result(self, params: np.ndarray | None,
                      output: np.ndarray | None,
                      score: float) -> dict:
        """Build a result dict from optimized parameters and output."""
        if params is None:
            params = np.zeros(self.domain.input_dim)
        if output is None:
            output = np.zeros(self.domain.output_dim)

        # Clip output for spectrum types
        if self.domain.output_type == "spectrum":
            output = np.clip(output, 0.0, 1.0)

        # Compute band means for spectrum
        band_means = {}
        if self.domain.output_type == "spectrum":
            n_bands = self.domain.critic_n_bands
            band_size = self.domain.output_dim // n_bands
            for i in range(n_bands):
                start = i * band_size
                end = min((i + 1) * band_size, self.domain.output_dim)
                band_means[f"band_{i}_mean"] = float(np.mean(output[start:end]))

        metrics = {
            "avg_output": float(np.mean(output)),
            "peak_output": float(np.max(output)),
            "min_output": float(np.min(output)),
            "std_output": float(np.std(output)),
            **band_means,
        }

        return {
            "optimized_params": params.tolist(),
            "output": output.tolist(),
            "spectrum": output.tolist(),   # Legacy compatibility
            "metrics": metrics,
            "best_score": float(score),
            "thickness_mm": None,          # Legacy compatibility
            "goal": self.domain.optimization_goal,
        }

    def reset_smoother(self):
        """Reset feedback smoother between independent optimization runs."""
        self._smoother.reset()
