"""Genetic Algorithm (NSGA-II style) baseline for inverse design."""

import numpy as np
import torch

from ...models.resnet_model import ResNetDNN
from ...tools.evaluate import evaluate_design
from ...utils.physics import N_PARAMS, PARAM_BOUNDS, THICKNESS_INDEX


def genetic_algorithm(
    model: ResNetDNN,
    target_thickness_mm: float,
    population_size: int = 100,
    generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism: int = 5,
    goal: str = "maximize_avg",
    device: torch.device | None = None,
    seed: int = 42,
) -> dict:
    """Genetic Algorithm baseline for metamaterial inverse design.

    Uses tournament selection, uniform crossover, and Gaussian mutation
    with projection onto physical bounds.

    Returns:
        dict with best design, metrics, and convergence history.
    """
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
            result = evaluate_design(model, population[i], device)
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
