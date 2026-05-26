"""Designer Agent: performs gradient-based inverse design optimization."""

import numpy as np
import torch

from .base_agent import BaseAgent, AgentMessage, AgentConfig


class DesignerAgent(BaseAgent):
    """The Designer Agent runs multi-start gradient optimization to find
    optimal metamaterial geometry parameters.

    It does NOT require an LLM — it's a deterministic optimization agent.
    The LLM is only used by the Orchestrator to decide when/how to invoke it.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 n_restarts: int = 5, steps_per_restart: int = 100):
        config = AgentConfig(
            name="Designer",
            role="Inverse Design Optimizer",
            system_prompt="You optimize metamaterial geometry for best absorption performance.",
        )
        super().__init__(config)
        self.model = model
        self.device = device
        self.n_restarts = n_restarts
        self.steps_per_restart = steps_per_restart

    def act(self, input_message: AgentMessage) -> AgentMessage:
        """Run multi-start gradient descent optimization.

        Expected input_message.content:
            - target_thickness_mm: float
            - optimization_goal: str ('maximize_avg' | 'maximize_peak')
            - init_strategy: str ('random' | 'default' | 'theory_guided')
            - frequency_targets: optional list of Hz targets for theory-guided init
            - critic_feedback: optional dict with spectral weaknesses to address
        """
        from ..tools.optimize import optimize_design
        from ..utils.physics import N_PARAMS, clamp_params

        content = input_message.content
        target_thickness = float(content.get("target_thickness_mm", 100.0))
        goal = content.get("optimization_goal", "maximize_avg")
        init_strategy = content.get("init_strategy", "default")
        critic_feedback = content.get("critic_feedback", None)

        best_result = None
        best_score = -np.inf

        for restart in range(self.n_restarts):
            # Generate initialization based on strategy
            init_params = self._generate_init(
                target_thickness, init_strategy, restart
            )

            # If critic identified weak bands, add weighted loss
            if critic_feedback and "weak_bands" in critic_feedback:
                result = self._optimize_weighted(
                    init_params, target_thickness, goal,
                    critic_feedback["weak_bands"]
                )
            else:
                result = optimize_design(
                    self.model,
                    target_thickness_mm=target_thickness,
                    goal=goal,
                    init_params=init_params,
                    steps=self.steps_per_restart,
                    device=self.device,
                    verbose=(restart == 0),
                )

            score = result["metrics"]["avg_absorption"]
            if score > best_score:
                best_score = score
                best_result = result

        best_result["n_restarts"] = self.n_restarts
        best_result["init_strategy"] = init_strategy

        return AgentMessage(
            sender=self.name,
            recipient=input_message.sender,
            content={"design": best_result},
            round_id=input_message.round_id,
        )

    def _generate_init(self, thickness: float, strategy: str,
                       restart_idx: int) -> np.ndarray:
        """Generate initial parameters based on strategy."""
        from ..utils.physics import PARAM_BOUNDS

        if strategy == "random":
            lo = PARAM_BOUNDS[:, 0]
            hi = PARAM_BOUNDS[:, 1]
            params = np.random.uniform(lo, hi)
            params[24] = thickness
            return params

        elif strategy == "theory_guided":
            # Cavity depth ~ quarter-wavelength at target frequency
            # Default to 500Hz: λ/4 ≈ 343/(4*500) ≈ 171.5mm for cavity depth
            base_params = np.full(31, 10.0)
            base_params[0:8] = np.random.uniform(5.0, 30.0, 8)  # Layer 1 cavities
            base_params[8:16] = np.random.uniform(10.0, 40.0, 8)  # Layer 2 cavities
            base_params[16:24] = np.random.uniform(5.0, 30.0, 8)  # Pores
            base_params[24] = thickness
            base_params[25:27] = np.random.uniform(10.0, 50.0, 2)
            base_params[27:31] = np.random.uniform(5.0, 30.0, 4)
            return base_params

        else:  # 'default'
            base = np.full(31, 10.0)
            base[24] = thickness
            # Add small random perturbation to break symmetry
            if restart_idx > 0:
                base += np.random.uniform(-2.0, 2.0, 31)
                base[24] = thickness
            return base

    def _optimize_weighted(self, init_params: np.ndarray, thickness: float,
                           goal: str, weak_bands: list[int]) -> dict:
        """Optimize with extra weight on bands the critic identified as weak."""
        import torch
        import torch.optim as optim
        from ..tools.optimize import optimize_design
        from ..utils.physics import PARAM_BOUNDS, THICKNESS_INDEX
        from ..utils.metrics import compute_all_metrics

        device = self.device
        param_tensor = torch.tensor(init_params, dtype=torch.float32,
                                     device=device, requires_grad=True)
        opt = optim.Adam([param_tensor], lr=0.5)

        best_score = -np.inf
        best_spectrum = None
        best_params = None
        n_freq = 100

        for step in range(self.steps_per_restart):
            opt.zero_grad()
            spectrum = self.model(param_tensor.unsqueeze(0)).flatten()

            if goal == "maximize_avg":
                base_loss = -torch.mean(spectrum)
            else:
                base_loss = -torch.max(spectrum)

            # Add extra penalty for weak bands
            band_size = n_freq // 5
            weak_penalty = 0.0
            for band_idx in weak_bands:
                start = band_idx * band_size
                end = start + band_size
                if start < n_freq:
                    band_mean = torch.mean(spectrum[start:end])
                    weak_penalty += -band_mean  # Maximize weak band

            loss = base_loss + 0.3 * weak_penalty

            loss.backward()
            opt.step()

            with torch.no_grad():
                param_tensor.data[THICKNESS_INDEX] = thickness
                lo = torch.tensor(PARAM_BOUNDS[:, 0], dtype=torch.float32, device=device)
                hi = torch.tensor(PARAM_BOUNDS[:, 1], dtype=torch.float32, device=device)
                param_tensor.data = torch.clamp(param_tensor.data, lo, hi)

            score = -loss.item()
            if score > best_score:
                best_score = score
                best_spectrum = spectrum.detach().cpu().numpy()
                best_params = param_tensor.detach().cpu().numpy().copy()

        best_spectrum = np.clip(best_spectrum, 0.0, 1.0)
        return {
            "optimized_params": best_params.tolist(),
            "spectrum": best_spectrum.tolist(),
            "metrics": compute_all_metrics(best_spectrum),
            "best_score": float(best_score),
            "goal": goal,
            "thickness_mm": thickness,
        }
