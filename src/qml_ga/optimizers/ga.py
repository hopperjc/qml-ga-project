import numpy as np
from packaging.version import Version
import pygad

def _pygad_fitness_wrapper(eval_solution_fn):
    """Cria função de fitness compatível com PyGAD 2.x e 3.x."""
    ver = Version(pygad.__version__)
    if ver < Version("3.0.0"):
        def fitness_v2(solution, sol_idx):
            return float(eval_solution_fn(solution))
        return fitness_v2
    else:
        def fitness_v3(ga_instance, solution, sol_idx):
            return float(eval_solution_fn(solution))
        return fitness_v3

def _normalize_alias(name: str, kind: str):
    name = (name or "").lower()
    if kind == "crossover":
        return "two_points" if name in {"two_points_crossover", "two_points"} else "single_point"
    if kind == "mutation":
        return "adaptive" if name in {"adaptive_mutation", "adaptive"} else "random"
    if kind == "selection":
        return "sss" if name in {"sss"} else "tournament"
    return name

def run_ga(
    eval_solution_fn,
    num_genes: int,
    num_generations: int,
    sol_per_pop: int,
    num_parents_mating: int,
    keep_parents: int,
    parent_selection_type: str,
    crossover_type: str,
    mutation_type: str,
    mutation_percent_genes: int,
    init_range_low: float = -3.14,
    init_range_high: float = 3.14,
    random_seed: int = 42,
):
    fitness_func = _pygad_fitness_wrapper(eval_solution_fn)

    parent_selection_type = _normalize_alias(parent_selection_type, "selection")
    crossover_type = _normalize_alias(crossover_type, "crossover")
    mutation_type = _normalize_alias(mutation_type, "mutation")

    ga = pygad.GA(
        num_generations=int(num_generations),
        sol_per_pop=int(sol_per_pop),
        num_parents_mating=int(num_parents_mating),
        num_genes=int(num_genes),
        init_range_low=float(init_range_low),
        init_range_high=float(init_range_high),
        parent_selection_type=parent_selection_type,
        keep_parents=int(keep_parents),
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=int(mutation_percent_genes),
        fitness_func=fitness_func,
        random_seed=int(random_seed),
    )
    ga.run()
    solution, fitness, sol_idx = ga.best_solution()
    return solution, float(fitness), int(sol_idx), ga
