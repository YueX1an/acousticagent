#!/usr/bin/env python
"""LLM-Guided Inverse Design — Main Entry Point.

Usage:
    python main.py --domain acoustic --critic llm
    python main.py --domain airfoil --critic heuristic   # ablation
    python main.py --benchmark                            # full suite
    python main.py --baseline ga --domain acoustic        # GA baseline only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from agent4science.domain_spec import load_domain, DomainSpec
from agent4science.models import ResNetDNN
from agent4science.agents.designer import Designer
from agent4science.agents.orchestrator import Orchestrator, OrchestratorConfig


# ── Model Loading ──

def load_surrogate(domain: DomainSpec, model_dir: str = "saved_models",
                   device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    """Load a pre-trained surrogate model for a domain.

    Tries: saved_models/surrogate_{domain.name}.pth first,
    then falls back to saved_models_dnn_v2/ResNet_PeakFocus.pth for acoustic.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetDNN(
        input_dim=domain.input_dim,
        output_dim=domain.output_dim,
        hidden_dim=domain.hidden_dim,
        num_blocks=domain.num_blocks,
    ).to(device)

    # Try domain-specific checkpoint first
    checkpoint_path = os.path.join(model_dir, f"surrogate_{domain.name}.pth")

    # Fallback for acoustic: use legacy checkpoint
    if not os.path.exists(checkpoint_path) and domain.name == "acoustic_metamaterial":
        legacy_path = os.path.join("saved_models_dnn_v2", "ResNet_PeakFocus.pth")
        if os.path.exists(legacy_path):
            checkpoint_path = legacy_path

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Model loaded: {checkpoint_path}")
    else:
        print(f"  [WARN] No checkpoint found at {checkpoint_path}")
        print(f"  Using RANDOM weights — results will not be physically meaningful.")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, device


# ── Critic Factory ──

def create_critic(domain: DomainSpec, critic_type: str = "llm",
                  model_name: str = "qwen2.5:14b",
                  backend: str = "ollama"):
    """Create a critic (LLM or Heuristic) for a domain."""
    if critic_type == "llm":
        from agent4science.agents.llm_critic import LLMCritic
        return LLMCritic(domain, model_name=model_name, backend=backend)
    elif critic_type == "heuristic":
        from agent4science.baselines.heuristic_critic import HeuristicCritic
        return HeuristicCritic(domain)
    else:
        raise ValueError(f"Unknown critic type: {critic_type}")


# ── Single Run ──

def run_design(domain_name: str, critic_type: str = "llm",
               constraint_value: float | None = None,
               n_restarts: int = 5, steps: int = 100,
               llm_model: str = "qwen2.5:14b",
               verbose: bool = True) -> dict:
    """Run a single multi-agent design optimization.

    Args:
        domain_name: 'acoustic', 'airfoil', or 'concrete'.
        critic_type: 'llm' or 'heuristic'.
        constraint_value: Value for constrained parameters.
        n_restarts, steps: Designer optimization settings.
        llm_model: LLM model identifier.
        verbose: Print progress.

    Returns:
        Result dict with final_design, summary, round_log.
    """
    domain = load_domain(domain_name)
    model, device = load_surrogate(domain)

    # Auto-detect constraint value for acoustic domain
    if constraint_value is None and domain.constrained_indices:
        # Use default from target_metrics
        constraint_value = 120.0  # Default for acoustic

    designer = Designer(
        model, domain, device=device,
        n_restarts=n_restarts, steps_per_restart=steps,
    )

    critic = create_critic(domain, critic_type, model_name=llm_model)

    config = OrchestratorConfig(max_rounds=5, convergence_threshold=0.005, verbose=verbose)
    orch = Orchestrator(designer, critic, domain, config)

    result = orch.run(constraint_value=constraint_value)

    # Save
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"results/{domain_name}_{critic_type}_{ts}.json"
    with open(path, "w") as f:
        json.dump(_serialize(result), f, indent=2)
    print(f"  Saved: {path}")

    return result


# ── Baseline Runner ──

def run_baseline(domain_name: str, method: str = "random_search",
                 constraint_value: float | None = None,
                 n_trials: int = 5000,
                 pop_size: int = 100, generations: int = 50,
                 bo_iters: int = 100) -> dict:
    """Run a baseline method on a domain."""
    domain = load_domain(domain_name)
    model, device = load_surrogate(domain)

    print(f"\n{'='*60}")
    print(f"  Baseline: {method} | Domain: {domain_name}")
    print(f"{'='*60}")

    t0 = time.time()

    if method == "random_search":
        from agent4science.experiments.baselines.random_search import random_search
        result = random_search(model, domain=domain, n_trials=n_trials, device=device)
    elif method == "genetic_algorithm":
        from agent4science.experiments.baselines.genetic import genetic_algorithm as ga_func
        result = ga_func(model, domain=domain, population_size=pop_size,
                         generations=generations, device=device)
    elif method == "bayesian_optimization":
        from agent4science.baselines.bayesian_opt import bayesian_optimization
        result = bayesian_optimization(
            model, domain, n_iterations=bo_iters, device=device,
        )
    else:
        raise ValueError(f"Unknown baseline: {method}")

    elapsed = time.time() - t0

    metrics = result.get("metrics", {})
    print(f"  Avg Output: {metrics.get('avg_output', 0):.4f}")
    print(f"  Peak Output: {metrics.get('peak_output', 0):.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"results/{domain_name}_{method}_{ts}.json"
    with open(path, "w") as f:
        json.dump(_serialize(result), f, indent=2)

    result["elapsed_s"] = elapsed
    return result


# ── Full Benchmark ──

def run_benchmark(domains: list[str] | None = None,
                  methods: list[str] | None = None,
                  seeds: int = 3):
    """Run full benchmark across domains and methods."""
    if domains is None:
        domains = ["acoustic", "airfoil", "concrete"]
    if methods is None:
        methods = ["llm", "heuristic", "designer_only",
                    "random_search", "genetic_algorithm",
                    "bayesian_optimization"]

    import csv

    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/benchmark_{ts}.csv"

    rows = []

    for domain_name in domains:
        for method in methods:
            print(f"\n{'#'*60}")
            print(f"#  Domain: {domain_name} | Method: {method}")
            print(f"{'#'*60}")

            for seed in range(seeds):
                np.random.seed(seed)
                torch.manual_seed(seed)

                try:
                    if method in ("llm", "heuristic"):
                        result = run_design(
                            domain_name, critic_type=method,
                            n_restarts=3, steps=100, verbose=(seed == 0),
                        )
                        row = {
                            "domain": domain_name, "method": method,
                            "seed": seed,
                            "avg_output": result["summary"].get("avg_output", 0),
                            "peak_output": result["summary"].get("peak_output", 0),
                            "rounds": result["total_rounds"],
                            "time_s": result["total_time_s"],
                            "verdict": result["verdict"],
                        }
                    elif method == "designer_only":
                        domain = load_domain(domain_name)
                        model, device = load_surrogate(domain)
                        designer = Designer(
                            model, domain, device=device,
                            n_restarts=3, steps_per_restart=100,
                        )
                        t0 = time.time()
                        design = designer.optimize(feedback=None)
                        elapsed = time.time() - t0
                        row = {
                            "domain": domain_name, "method": method,
                            "seed": seed,
                            "avg_output": design["metrics"].get("avg_output", 0),
                            "peak_output": design["metrics"].get("peak_output", 0),
                            "rounds": 1, "time_s": elapsed, "verdict": "n/a",
                        }
                    else:
                        result = run_baseline(domain_name, method)
                        row = {
                            "domain": domain_name, "method": method,
                            "seed": seed,
                            "avg_output": result["metrics"].get("avg_output", 0),
                            "peak_output": result["metrics"].get("peak_output", 0),
                            "rounds": 1,
                            "time_s": result.get("elapsed_s", 0),
                            "verdict": "n/a",
                        }

                    rows.append(row)

                    if seed == 0:
                        print(f"  avg_output={row['avg_output']:.4f} | "
                              f"time={row['time_s']:.1f}s")

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    import traceback
                    traceback.print_exc()
                    rows.append({
                        "domain": domain_name, "method": method,
                        "seed": seed, "error": str(e),
                    })

    # Write CSV
    if rows:
        fieldnames = ["domain", "method", "seed", "avg_output",
                       "peak_output", "rounds", "time_s", "verdict"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[SAVED] {csv_path}")

        # Summary table
        print(f"\n{'Domain':<16} {'Method':<24} {'Avg':>8} {'Time':>8}")
        print("-" * 60)
        from collections import defaultdict
        agg = defaultdict(dict)
        for r in rows:
            if "error" in r:
                continue
            key = (r["domain"], r["method"])
            if key not in agg:
                agg[key] = {"avgs": [], "times": []}
            agg[key]["avgs"].append(r.get("avg_output", 0))
            agg[key]["times"].append(r.get("time_s", 0))

        for (dom, meth), vals in sorted(agg.items()):
            avg = np.mean(vals["avgs"])
            t = np.mean(vals["times"])
            print(f"{dom:<16} {meth:<24} {avg:>8.4f} {t:>7.1f}s")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided Inverse Design — Domain-Agnostic Framework"
    )
    parser.add_argument("--domain", type=str, default="acoustic",
                        choices=["acoustic", "airfoil", "concrete"],
                        help="Design domain")
    parser.add_argument("--critic", type=str, default="llm",
                        choices=["llm", "heuristic"],
                        help="Critic type: llm (proposed) or heuristic (ablation)")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["random_search", "genetic_algorithm",
                                  "bayesian_optimization"],
                        help="Run a specific baseline")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark suite")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:14b",
                        help="LLM model name")
    parser.add_argument("--constraint", type=float, default=None,
                        help="Constraint value for locked parameters")
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LLM-Guided Inverse Design Framework")
    print("  Domain-Agnostic | Cross-Physics Validation")
    print("=" * 60 + "\n")

    if args.benchmark:
        run_benchmark(seeds=args.seeds)
    elif args.baseline:
        run_baseline(args.domain, args.baseline, constraint_value=args.constraint)
    else:
        run_design(
            args.domain,
            critic_type=args.critic,
            constraint_value=args.constraint,
            n_restarts=args.restarts,
            steps=args.steps,
            llm_model=args.llm_model,
        )


def _serialize(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
