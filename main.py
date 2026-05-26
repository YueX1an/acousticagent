#!/usr/bin/env python
"""Main entry point for the Multi-Agent Scientific Inverse Design system.

Usage:
    python main.py                          # Run interactive multi-agent design
    python main.py --task broadband         # Run a specific design task
    python main.py --benchmark              # Run all benchmarks
    python main.py --baseline random_search --task broadband  # Run baseline
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from agent4science.models import ResNetDNN
from agent4science.agents.designer_agent import DesignerAgent
from agent4science.agents.critic_agent import CriticAgent
from agent4science.agents.orchestrator import Orchestrator, OrchestratorConfig
from agent4science.tools.visualize import plot_spectrum, plot_multi_agent_progress
from agent4science.utils.metrics import compute_all_metrics

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = os.path.join("saved_models_dnn_v2", "ResNet_PeakFocus.pth")
MODEL_ARGS = {
    "input_dim": 31,
    "output_dim": 100,
    "hidden_dim": 1024,
    "num_blocks": 8,
}

TASKS = {
    "broadband": {
        "target_thickness_mm": 120,
        "optimization_goal": "maximize_avg",
        "init_strategy": "default",
        "target_spec": {
            "min_avg_absorption": 0.70,
            "min_peak_absorption": 0.90,
            "min_band_absorption": 0.60,
            "min_bandwidth_0.8": 400,
        },
    },
    "low_frequency": {
        "target_thickness_mm": 150,
        "optimization_goal": "maximize_peak",
        "init_strategy": "theory_guided",
        "target_spec": {
            "min_avg_absorption": 0.50,
            "min_peak_absorption": 0.85,
            "min_band_absorption": 0.40,
            "min_bandwidth_0.8": 200,
        },
    },
    "multi_band": {
        "target_thickness_mm": 120,
        "optimization_goal": "maximize_avg",
        "init_strategy": "theory_guided",
        "target_spec": {
            "min_avg_absorption": 0.60,
            "min_peak_absorption": 0.85,
            "min_band_absorption": 0.50,
            "min_bandwidth_0.8": 300,
        },
    },
    "ultra_thin": {
        "target_thickness_mm": 50,
        "optimization_goal": "maximize_avg",
        "init_strategy": "random",
        "target_spec": {
            "min_avg_absorption": 0.40,
            "min_peak_absorption": 0.75,
            "min_band_absorption": 0.30,
            "min_bandwidth_0.8": 150,
        },
    },
}


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model(device: torch.device | None = None) -> ResNetDNN:
    """Load the pre-trained ResNetDNN surrogate model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading surrogate model on {device}...")
    model = ResNetDNN(**MODEL_ARGS)

    if os.path.exists(DEFAULT_MODEL_PATH):
        state_dict = torch.load(DEFAULT_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Model loaded from: {DEFAULT_MODEL_PATH}")
    else:
        print(f"  [WARN] Model file not found: {DEFAULT_MODEL_PATH}")
        print("  Using RANDOM weights (demonstration mode)")

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, device


# ─── Multi-Agent Design ──────────────────────────────────────────────────────

def run_multi_agent(task_id: str = "broadband", verbose: bool = True) -> dict:
    """Run the multi-agent design loop for a given task."""
    task = TASKS.get(task_id, TASKS["broadband"])

    model, device = load_model()

    designer = DesignerAgent(
        model=model,
        device=device,
        n_restarts=5,
        steps_per_restart=100,
    )

    critic = CriticAgent()

    config = OrchestratorConfig(
        max_rounds=5,
        convergence_threshold=0.005,
        verbose=verbose,
        **{k: task[k] for k in ("target_thickness_mm", "optimization_goal",
                                  "init_strategy", "target_spec")},
    )

    orchestrator = Orchestrator(designer, critic, config)
    result = orchestrator.run(user_task=task)

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = f"results/{task_id}_{timestamp}.json"

    # Convert numpy arrays for JSON serialization
    serializable = _make_serializable(result)
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved: {result_path}")

    # Plot
    spectrum = np.array(result["final_design"].get("spectrum", []))
    if len(spectrum) > 0:
        plot_path = f"results/{task_id}_{timestamp}_spectrum.png"
        plot_spectrum(spectrum, title=f"Multi-Agent Design: {task_id}",
                       save_path=plot_path, show=False)

    if result["round_log"]:
        progress_path = f"results/{task_id}_{timestamp}_progress.png"
        plot_multi_agent_progress(result["round_log"],
                                   save_path=progress_path, show=False)

    return result


# ─── Baselines ───────────────────────────────────────────────────────────────

def run_baseline(task_id: str, method: str = "random_search") -> dict:
    """Run a baseline method on a given task."""
    from agent4science.experiments.baselines.random_search import random_search
    from agent4science.experiments.baselines.genetic import genetic_algorithm

    task = TASKS.get(task_id, TASKS["broadband"])
    model, device = load_model()

    print(f"\nRunning {method} baseline for task: {task_id}")

    if method == "random_search":
        result = random_search(
            model, task["target_thickness_mm"],
            n_trials=5000, goal=task["optimization_goal"],
            device=device,
        )
    elif method == "genetic_algorithm":
        result = genetic_algorithm(
            model, task["target_thickness_mm"],
            population_size=100, generations=50,
            goal=task["optimization_goal"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = f"results/{task_id}_{method}_{timestamp}.json"

    serializable = _make_serializable(result)
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved: {result_path}")

    return result


# ─── Full Benchmark ──────────────────────────────────────────────────────────

def run_benchmark():
    """Run full benchmark: multi-agent + baselines on all tasks."""
    import csv

    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/benchmark_{timestamp}.csv"

    results = []
    methods = ["multi_agent", "random_search", "genetic_algorithm"]

    for task_id in TASKS:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"  Benchmark: {task_id} | {method}")
            print(f"{'='*60}")

            try:
                if method == "multi_agent":
                    result = run_multi_agent(task_id, verbose=True)
                    metrics = result["summary"]
                elif method == "random_search":
                    result = run_baseline(task_id, method)
                    metrics = result["metrics"]
                elif method == "genetic_algorithm":
                    result = run_baseline(task_id, method)
                    metrics = result["metrics"]

                results.append({
                    "task": task_id,
                    "method": method,
                    "avg_absorption": metrics.get("avg_absorption", 0),
                    "peak_absorption": metrics.get("peak_absorption", 0),
                    "bandwidth_0.8_hz": metrics.get("bandwidth_0.8_hz", 0),
                    "worst_band_absorption": metrics.get("worst_band_absorption", 0),
                })
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append({
                    "task": task_id,
                    "method": method,
                    "error": str(e),
                })

    # Write CSV
    if results:
        fieldnames = ["task", "method", "avg_absorption", "peak_absorption",
                       "bandwidth_0.8_hz", "worst_band_absorption"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[SAVED] Benchmark results: {csv_path}")

        # Print summary table
        print(f"\n{'Task':<16} {'Method':<22} {'Avg Abs':>8} {'Peak Abs':>8} {'BW 0.8':>8} {'Worst Band':>10}")
        print("-" * 80)
        for r in results:
            if "error" not in r:
                print(f"{r['task']:<16} {r['method']:<22} "
                      f"{r['avg_absorption']:>8.4f} {r['peak_absorption']:>8.4f} "
                      f"{r['bandwidth_0.8_hz']:>8.0f} {r['worst_band_absorption']:>10.4f}")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _make_serializable(obj):
    """Convert numpy arrays and other non-serializable types for JSON export."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Scientific Inverse Design for Acoustic Metamaterials"
    )
    parser.add_argument("--task", type=str, default="broadband",
                        choices=list(TASKS.keys()),
                        help="Design task to run (default: broadband)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark suite")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["random_search", "genetic_algorithm"],
                        help="Run a specific baseline method")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")

    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Multi-Agent Inverse Design for Acoustic Metamaterials  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    if args.benchmark:
        run_benchmark()
    elif args.baseline:
        run_baseline(args.task, args.baseline)
    else:
        run_multi_agent(args.task)


if __name__ == "__main__":
    main()
