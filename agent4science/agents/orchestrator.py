"""Orchestrator: convergence-gated refinement loop.

The Orchestrator is domain-agnostic control logic — NOT an AI agent.
It coordinates: Designer → LLM Critic (or Heuristic Critic) → repeat until pass or stall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain_spec import DomainSpec
from ..text_to_math import RefinementFeedback


@dataclass
class OrchestratorConfig:
    max_rounds: int = 5
    convergence_threshold: float = 0.005
    verbose: bool = True


class Orchestrator:
    """Coordinates the Designer↔Critic refinement loop.

    Flow:
    1. Designer optimizes → produces design
    2. Critic reviews → returns RefinementFeedback (verdict + weak_regions)
    3. If "pass": done
    4. If "refine": Designer re-optimizes with feedback
    5. If stalled: stop early
    """

    def __init__(self, designer, critic, domain: DomainSpec,
                 config: OrchestratorConfig | None = None):
        """
        Args:
            designer: Designer instance (algorithmic optimizer).
            critic: LLMCritic or HeuristicCritic instance.
            domain: DomainSpec.
            config: Orchestrator configuration.
        """
        self.designer = designer
        self.critic = critic
        self.domain = domain
        self.config = config or OrchestratorConfig()

        self._round_log: list[dict] = []

    def run(self, constraint_value: float | None = None,
            use_minimal_prompt: bool = False) -> dict:
        """Execute the full refinement loop.

        Args:
            constraint_value: Value for constrained parameters.
            use_minimal_prompt: If True, use minimal prompt (A3 ablation).

        Returns:
            dict with: final_design, verdict, total_rounds, total_time_s,
                round_log, summary.
        """
        cv = self.config
        if cv.verbose:
            print(f"\n{'='*60}")
            print(f"  LLM-Guided Inverse Design: {self.domain.name}")
            print(f"  Output: {self.domain.output_type} | Goal: {self.domain.optimization_goal}")
            print(f"  Max rounds: {cv.max_rounds} | Threshold: {cv.convergence_threshold}")
            print(f"{'='*60}\n")

        t_start = time.time()
        critic_feedback: RefinementFeedback | None = None
        final_verdict = "refine"
        final_design = None

        # Reset smoother for independent run
        if hasattr(self.designer, 'reset_smoother'):
            self.designer.reset_smoother()

        for round_id in range(1, cv.max_rounds + 1):
            if cv.verbose:
                print(f"--- Round {round_id}/{cv.max_rounds} ---")
                print(f"  [Designer] Optimizing...")

            t_round_start = time.time()

            # Phase 1: Designer optimizes
            design = self.designer.optimize(
                feedback=critic_feedback,
                constraint_value=constraint_value,
            )

            t_design = time.time() - t_round_start

            metrics = design.get("metrics", {})
            if cv.verbose:
                avg = metrics.get("avg_output", 0.0)
                peak = metrics.get("peak_output", 0.0)
                print(f"  [Designer] Done in {t_design:.1f}s | avg_output={avg:.4f} | peak={peak:.4f}")

            # Phase 2: Critic reviews
            if cv.verbose:
                print(f"  [Critic] Reviewing design...")

            critic_feedback = self.critic.review(
                design_params=design.get("optimized_params", []),
                design_metrics=metrics,
                constraint_value=constraint_value,
                use_minimal_prompt=use_minimal_prompt,
            )

            if cv.verbose:
                n_weak = len(critic_feedback.weak_regions)
                n_violations = len(critic_feedback.physical_violations)
                print(f"  [Critic] Verdict: {critic_feedback.verdict.upper()} | "
                      f"Weak regions: {n_weak} | Violations: {n_violations}")

            # Check convergence
            if critic_feedback.is_pass:
                final_verdict = "pass"
                final_design = design
                if cv.verbose:
                    print(f"\n[OK] Design accepted at round {round_id}!")
                break

            # Check stalling
            if len(self._round_log) > 0:
                prev_avg = self._round_log[-1].get("best_avg", 0.0)
                curr_avg = metrics.get("avg_output", 0.0)
                delta = abs(curr_avg - prev_avg)
                if delta < cv.convergence_threshold:
                    if cv.verbose:
                        print(f"  [WARN] Improvement stalled (delta={delta:.5f})")
                    if len(self._round_log) >= 2:
                        final_verdict = "stalled"
                        final_design = design
                        if cv.verbose:
                            print("  Stopping: no significant improvement for 2 rounds")
                        break

            self._round_log.append({
                "round": round_id,
                "best_avg": metrics.get("avg_output", 0.0),
                "best_peak": metrics.get("peak_output", 0.0),
                "verdict": critic_feedback.verdict,
                "n_weak_regions": len(critic_feedback.weak_regions),
                "n_violations": len(critic_feedback.physical_violations),
            })

            final_design = design
            final_verdict = critic_feedback.verdict

        if final_design is None:
            # If optimization produced nothing, run one last round
            final_design = self.designer.optimize(
                feedback=None, constraint_value=constraint_value
            )
            final_verdict = "no_critic"

        t_end = time.time()

        result = {
            "final_design": final_design,
            "verdict": final_verdict,
            "total_rounds": len(self._round_log),
            "total_time_s": round(t_end - t_start, 1),
            "round_log": self._round_log,
            "summary": {
                "avg_output": final_design["metrics"].get("avg_output", 0.0),
                "peak_output": final_design["metrics"].get("peak_output", 0.0),
                "optimization_goal": self.domain.optimization_goal,
                "domain": self.domain.name,
            },
        }

        if cv.verbose:
            summary = result["summary"]
            print(f"\n{'='*60}")
            print(f"  Design Complete: {self.domain.name}")
            print(f"  Rounds: {len(self._round_log)} | Time: {t_end - t_start:.1f}s")
            print(f"  Avg Output: {summary['avg_output']:.4f}")
            print(f"  Verdict: {final_verdict}")
            print(f"{'='*60}\n")

        return result
