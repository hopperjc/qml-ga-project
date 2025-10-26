from typing import Callable, Tuple, Optional
import numpy as np
import pygad
from qml_ga.utils.logger import append_csv_row

ProgressLogger = Optional[Callable[[str], None]]

def run_ga(
    eval_solution_fn: Callable[[np.ndarray], float],
    num_genes: int,
    num_generations: int,
    sol_per_pop: int,
    num_parents_mating: int,
    keep_parents: int,
    parent_selection_type: str,
    crossover_type: str,
    mutation_type: str,
    mutation_percent_genes: int,
    init_range_low: float,
    init_range_high: float,
    random_seed: int = 42,
    # logging
    progress_logger: ProgressLogger = None,
    progress_csv_path: Optional[str] = None,
    log_every: int = 1,
    # métricas opcionais com melhor solução da geração
    metrics_from_solution: Optional[Callable[[np.ndarray], dict]] = None,
) -> Tuple[np.ndarray, float, int, pygad.GA]:
    """
    Executa GA com PyGAD e registra progresso por geração.
    metrics_from_solution: recebe o vetor de solução (genes) e devolve um dict com métricas adicionais para log.
    """
    def fitness_func(ga_inst, solution, solution_idx):
        return eval_solution_fn(solution)

    def on_generation(ga_inst: pygad.GA):
        gen = ga_inst.generations_completed
        if gen % max(1, log_every) != 0:
            return
        sol, fit, idx = ga_inst.best_solution()
        msg = f"gen {gen}/{num_generations} best_fitness={float(fit):.6f}"
        extra = {}
        if metrics_from_solution is not None:
            try:
                extra = metrics_from_solution(sol) or {}
                if extra:
                    msg += " " + " ".join([f"{k}={v:.4f}" if isinstance(v,(int,float)) else f"{k}={v}" for k,v in extra.items()])
            except Exception:
                pass
        if progress_logger:
            progress_logger(msg)
        if progress_csv_path:
            row = {"phase": "ga_gen", "generation": gen, "best_fitness": float(fit)}
            row.update({f"best_{k}": v for k, v in extra.items()})
            append_csv_row(progress_csv_path, row)

    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        random_seed=random_seed,
        on_generation=on_generation,
    )

    ga.run()
    solution, fitness, solution_idx = ga.best_solution()
    return np.array(solution), float(fitness), int(solution_idx), ga
