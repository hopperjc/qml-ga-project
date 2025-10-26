import numpy as np
import pygad

def _map_selection(sel: str) -> str:
    s = sel.lower()
    if s in ("steady-state", "sss"): return "sss"
    return s  # "tournament", "rank", "rws", etc.

class GAOptimizer:
    def __init__(self, vqc, ansatz, ga_cfg, X_train, y_train, X_val=None, y_val=None):
        self.vqc, self.ansatz = vqc, ansatz
        self.cfg = ga_cfg or {}
        self.Xt, self.yt = X_train, y_train
        self.Xv, self.yv = X_val, y_val
        self.n_params = ansatz.num_params(vqc.dev.num_wires)

    def _fitness(self, solution, _):
        y_pred = self.vqc.predict(solution, self.Xt)
        return (y_pred == self.yt).mean()

    def run(self):
        gene_space = {"low": -np.pi, "high": np.pi}
        ga = pygad.GA(
            num_generations=self.cfg.get("num_generations", 50),
            num_parents_mating=self.cfg.get("num_parents_mating", 8),
            fitness_func=self._fitness,
            sol_per_pop=self.cfg.get("population_size", 30),
            num_genes=self.n_params,
            gene_space=gene_space,
            parent_selection_type=_map_selection(self.cfg.get("selection_type", "tournament")),
            crossover_type=self.cfg.get("crossover_type", "single_point"),
            mutation_type=self.cfg.get("mutation_type", "random"),
            mutation_probability=self.cfg.get("mutation_probability", 0.1),
            crossover_probability=self.cfg.get("crossover_probability", 0.8),
            keep_elitism=self.cfg.get("elitism", 2),
            stop_criteria=self.cfg.get("stop_criteria", None),
            allow_duplicate_genes=True,
            suppress_warnings=True,
        )
        ga.run()
        best, best_fit, _ = ga.best_solution()
        history = [float(v) for v in ga.best_solutions_fitness]
        return np.array(best), float(best_fit), history
