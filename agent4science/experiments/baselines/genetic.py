"""Genetic Algorithm (NSGA-II style) baseline for inverse design.

Supports both legacy (hardcoded acoustic) and DomainSpec-based interfaces.
"""

from __future__ import annotations

import numpy as np
import torch

from ...models.resnet_model import ResNetDNN
from ...tools.evaluate import evaluate_design
from ...utils.physics import N_PARAMS, PARAM_BOUNDS, THICKNESS_INDEX
from ...domain_spec import DomainSpec


def genetic_algorithm(
    model: ResNetDNN,
    target_thickness_mm: float | None = None,
    domain: DomainSpec | None = None,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: int = 5,
    goal: str = "maximize_avg",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Genetic Algorithm baseline.

    Can be called with DomainSpec (preferred) or legacy acoustic args.

    Args:
        model: Pre-trained ResNetDNN surrogate model.
        target_thickness_mm: Target thickness (legacy, acoustic-only).
        domain: DomainSpec (preferred interface).
        population_size: GA population size.
        generations: Number of generations.
        mutation_rate: Gaussian mutation probability per gene.
        crossover_rate: Uniform crossover probability.
        elitism: Number of elite individuals preserved.
        goal: Optimization goal.
        device: Torch device.
        seed: Random seed.

    Returns:
        dict with best design, metrics, and convergence history.
    """
    if domain is not None:
        return _genetic_algorithm_domain(
            model, domain, population_size, generations,
            mutation_rate, crossover_rate, elitism, device, seed,
        )
    return _genetic_algorithm_legacy(
        model, target_thickness_mm, population_size, generations,
        mutation_rate, crossover_rate, elitism, goal, device, seed,
    )


def _genetic_algorithm_domain(
    model, domain, pop_size, gens, mut_rate, cross_rate, elit, device, seed,
) -> dict:
    """GA with DomainSpec."""
    rng = np.random.RandomState(seed)
    bounds = domain.bounds_array
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    input_dim = domain.input_dim

    population = rng.uniform(lo, hi, size=(pop_size, input_dim))
    for idx in domain.constrained_indices:
        population[:, idx] = lo[idx]

    history = []

    for gen in range(gens):
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            result = evaluate_design(model, population[i], device, output_type=domain.output_type)
            output = np.array(result["spectrum"])
            if domain.optimization_goal in ("maximize_avg", "maximize_output"):
                fitness[i] = float(np.mean(output))
            elif domain.optimization_goal == "maximize_peak":
                fitness[i] = float(np.max(output))
            elif domain.optimization_goal == "maximize_ratio":
                fitness[i] = float(output[0] / max(output[1], 1e-6)) if len(output) >= 2 else float(np.mean(output))
            else:
                fitness[i] = float(np.mean(output))

        elite_idx = np.argsort(fitness)[::-1]
        history.append({"generation": gen, "best_fitness": float(fitness[elite_idx[0]]),
                         "mean_fitness": float(np.mean(fitness))})

        new_pop = np.zeros_like(population)
        new_pop[:elit] = population[elite_idx[:elit]]

        for i in range(elit, pop_size):
            t1, t2 = rng.choice(pop_size, size=2, replace=False)
            p1 = population[t1] if fitness[t1] > fitness[t2] else population[t2]
            if rng.rand() < cross_rate:
                t1, t2 = rng.choice(pop_size, size=2, replace=False)
                p2 = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                mask = rng.rand(input_dim) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            mut_mask = rng.rand(input_dim) < mut_rate
            child[mut_mask] += rng.randn(np.sum(mut_mask)) * (hi - lo)[mut_mask] * 0.1
            child = np.clip(child, lo, hi)
            for idx in domain.constrained_indices:
                child[idx] = lo[idx]
            new_pop[i] = child

        population = new_pop

    best_idx = np.argmax(fitness)
    best_result = evaluate_design(model, population[best_idx], device)

    return {
        "best_params": population[best_idx].tolist(),
        "best_spectrum": np.array(best_result["spectrum"]).tolist(),
        "metrics": {"avg_output": float(np.mean(best_result["spectrum"])),
                     "peak_output": float(np.max(best_result["spectrum"]))},
        "best_fitness": float(fitness[best_idx]),
        "convergence_history": history,
        "config": {"population_size": pop_size, "generations": gens,
                    "mutation_rate": mut_rate, "crossover_rate": cross_rate},
    }


def _genetic_algorithm_legacy(
    model: ResNetDNN,
    target_thickness_mm: float | None = None,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: int = 5,
    goal: str = "maximize_avg",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Legacy GA for acoustic domain (hardcoded 31-dim params)."""
    rng = np.random.RandomState(seed)
    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]

    # Initialize population
    population = rng.uniform(lo, hi, size=(population_size, N_PARAMS))
    population[:, THICKNESS_INDEX] = target_thickness_mm

    history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness = np.zeros(population_size)
        for i in range(population_size):
            result = evaluate_design(model, population[i], device, output_type='spectrum')
            spectrum = np.array(result["spectrum"])
            if goal == "maximize_avg":
                fitness[i] = float(np.mean(spectrum))
            elif goal == "maximize_peak":
                fitness[i] = float(np.max(spectrum))
            else:
                fitness[i] = float(np.mean(spectrum))

        # Sort by fitness (descending)
        elite_indices = np.argsort(fitness)[::-1]
        elite_fitness = fitness[elite_indices]

        history.append({
            "generation": gen,
            "best_fitness": float(elite_fitness[0]),
            "mean_fitness": float(np.mean(fitness)),
        })

        # Create next generation
        new_population = np.zeros_like(population)

        # Elitism: keep best individuals
        new_population[:elitism] = population[elite_indices[:elitism]]

        # Fill rest with selection, crossover, mutation
        for i in range(elitism, population_size):
            # Tournament selection (tournament size = 3)
            t1, t2 = rng.choice(population_size, size=2, replace=False)
            parent1 = population[t1] if fitness[t1] > fitness[t2] else population[t2]

            if rng.rand() < crossover_rate:
                t1, t2 = rng.choice(population_size, size=2, replace=False)
                parent2 = population[t1] if fitness[t1] > fitness[t2] else population[t2]
                # Uniform crossover
                mask = rng.rand(N_PARAMS) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()

            # Gaussian mutation
            mutation_mask = rng.rand(N_PARAMS) < mutation_rate
            child[mutation_mask] += rng.randn(np.sum(mutation_mask)) * 3.0

            # Project onto bounds
            child = np.clip(child, lo, hi)
            child[THICKNESS_INDEX] = target_thickness_mm

            new_population[i] = child

        population = new_population

    # Get best from final generation
    best_idx = np.argmax(fitness)
    best_params = population[best_idx]
    best_result = evaluate_design(model, best_params, device)
    best_spectrum = np.array(best_result["spectrum"])

    return {
        "best_params": best_params.tolist(),
        "best_spectrum": best_spectrum.tolist(),
        "metrics": best_result["metrics"],
        "best_fitness": float(fitness[best_idx]),
        "convergence_history": history,
        "config": {
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
        },
    }
