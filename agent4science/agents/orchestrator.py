"""Orchestrator: coordinates multi-agent debate and refinement loops."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base_agent import AgentMessage
from .designer_agent import DesignerAgent
from .critic_agent import CriticAgent


@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""

    max_rounds: int = 5
    convergence_threshold: float = 0.005  # Min avg_absorption improvement to continue
    target_thickness_mm: float = 100.0
    optimization_goal: str = "maximize_avg"
    init_strategy: str = "default"
    verbose: bool = True
    target_spec: dict = field(default_factory=lambda: {
        "min_avg_absorption": 0.7,
        "min_peak_absorption": 0.9,
        "min_band_absorption": 0.6,
        "min_bandwidth_0.8": 400,
    })


class Orchestrator:
    """Coordinates the multi-agent scientific design loop.

    Flow:
    1. Orchestrator sends design task → Designer
    2. Designer runs multi-start gradient optimization → returns best design
    3. Orchestrator sends design → Critic
    4. Critic validates and identifies weaknesses → returns critique
    5. If critique says "refine": Orchestrator sends feedback → Designer (go to 2)
    6. If critique says "pass": convergence achieved, return final design
    """

    def __init__(self, designer: DesignerAgent, critic: CriticAgent,
                 config: OrchestratorConfig | None = None):
        self.designer = designer
        self.critic = critic
        self.config = config or OrchestratorConfig()

        self._round_log: list[dict] = []
        self._conversation: list[AgentMessage] = []

    def run(self, user_task: dict | None = None) -> dict:
        """Execute the full multi-agent design loop.

        Args:
            user_task: dict with keys:
                - target_thickness_mm: float
                - optimization_goal: str
                - init_strategy: str
                - target_spec: dict (optional, overrides config)

        Returns:
            dict with: final_design, conversation_log, convergence, stats
        """
        user_task = user_task or {}
        target_thickness = user_task.get(
            "target_thickness_mm", self.config.target_thickness_mm
        )
        goal = user_task.get("optimization_goal", self.config.optimization_goal)
        init_strategy = user_task.get("init_strategy", self.config.init_strategy)
        target_spec = user_task.get("target_spec", self.config.target_spec)

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"  Multi-Agent Inverse Design Loop")
            print(f"  Target: {target_thickness}mm | Goal: {goal}")
            print(f"  Strategy: {init_strategy} | Max Rounds: {self.config.max_rounds}")
            print(f"{'='*60}\n")

        t_start = time.time()
        critic_feedback = None
        final_verdict = None

        for round_id in range(1, self.config.max_rounds + 1):
            if self.config.verbose:
                print(f"--- Round {round_id}/{self.config.max_rounds} ---")

            # Phase 1: Designer optimizes
            if self.config.verbose:
                print(f"  [Designer] Running multi-start optimization...")

            design_msg = AgentMessage(
                sender="Orchestrator",
                recipient="Designer",
                content={
                    "task": "Optimize metamaterial design",
                    "target_thickness_mm": target_thickness,
                    "optimization_goal": goal,
                    "init_strategy": init_strategy,
                    "critic_feedback": critic_feedback,
                    "target_spec": target_spec,
                },
                round_id=round_id,
            )

            t_design_start = time.time()
            designer_response = self.designer.act(design_msg)
            t_design_end = time.time()

            self._conversation.append(design_msg)
            self._conversation.append(designer_response)

            design = designer_response.content.get("design", {})
            if self.config.verbose:
                metrics = design.get("metrics", {})
                print(f"  [Designer] Done in {t_design_end - t_design_start:.1f}s")
                print(f"    Avg: {metrics.get('avg_absorption', 0):.3f} | "
                      f"Peak: {metrics.get('peak_absorption', 0):.3f}")

            # Phase 2: Critic evaluates
            if self.config.verbose:
                print(f"  [Critic] Evaluating design...")

            critique_msg = AgentMessage(
                sender="Orchestrator",
                recipient="Critic",
                content={
                    "task": "Validate and critique this design",
                    "design": design,
                    "target_spec": target_spec,
                },
                round_id=round_id,
            )

            critic_response = self.critic.act(critique_msg)
            self._conversation.append(critique_msg)
            self._conversation.append(critic_response)

            critique = critic_response.content
            verdict = critique.get("verdict", "refine")
            final_verdict = verdict

            if self.config.verbose:
                weak_n = len(critique.get("weak_bands", []))
                print(f"  [Critic] Verdict: {verdict.upper()} | "
                      f"Weak bands: {weak_n} | "
                      f"Targets met: {critique.get('targets_met', False)}")

            # Check convergence
            if verdict == "pass":
                if self.config.verbose:
                    print(f"\n✅ Convergence achieved at round {round_id}!")
                break

            # Prepare critic feedback for next round
            critic_feedback = {
                "weak_bands": critique.get("weak_bands", []),
                "suggestions": critique.get("suggestions", []),
                "failures": critique.get("failures", []),
            }

            # Check if improvement is stalling
            if len(self._round_log) > 0:
                prev_best = self._round_log[-1].get("best_avg", 0)
                curr_best = design.get("metrics", {}).get("avg_absorption", 0)
                if abs(curr_best - prev_best) < self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"  ⚠️ Improvement stalled (delta={curr_best - prev_best:.5f})")
                    # Force one more round then stop
                    if len(self._round_log) >= 2:
                        if self.config.verbose:
                            print("  Stopping: no significant improvement in 2 rounds")
                        final_verdict = "stalled"
                        break

            self._round_log.append({
                "round": round_id,
                "best_avg": design.get("metrics", {}).get("avg_absorption", 0),
                "best_peak": design.get("metrics", {}).get("peak_absorption", 0),
                "verdict": verdict,
                "weak_bands": critique.get("weak_bands", []),
            })

        t_end = time.time()

        # Compile final results
        final_design = designer_response.content.get("design", {})
        all_metrics = final_design.get("metrics", {})

        result = {
            "final_design": final_design,
            "verdict": final_verdict,
            "total_rounds": len(self._round_log),
            "total_time_s": round(t_end - t_start, 1),
            "conversation_log": self._format_conversation(),
            "round_log": self._round_log,
            "summary": {
                "avg_absorption": all_metrics.get("avg_absorption", 0),
                "peak_absorption": all_metrics.get("peak_absorption", 0),
                "bandwidth_0.8_hz": all_metrics.get("bandwidth_0.8_hz", 0),
                "worst_band_absorption": all_metrics.get("worst_band_absorption", 0),
                "thickness_mm": target_thickness,
                "optimization_goal": goal,
            },
        }

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"  Multi-Agent Design Complete")
            print(f"  Rounds: {len(self._round_log)} | Time: {t_end - t_start:.1f}s")
            print(f"  Final Avg Absorption: {all_metrics.get('avg_absorption', 0):.4f}")
            print(f"  Verdict: {final_verdict}")
            print(f"{'='*60}\n")

        return result

    def _format_conversation(self) -> list[dict]:
        """Format conversation log for export."""
        return [
            {
                "sender": msg.sender,
                "recipient": msg.recipient,
                "round": msg.round_id,
                "content_summary": {
                    k: v for k, v in msg.content.items()
                    if k in ("verdict", "targets_met", "weak_bands",
                              "optimization_goal", "target_thickness_mm")
                },
            }
            for msg in self._conversation
        ]
