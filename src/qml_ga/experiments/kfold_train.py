from typing import Dict, Any, List
import os, json
import numpy as np
import pennylane.numpy as qnp
from sklearn.model_selection import StratifiedKFold

from qml_ga.utils.io import ensure_dirs
from qml_ga.data.datamodule import DataModule
from qml_ga.circuits.vqc import build_vqc
from qml_ga.ansatz.base import init_weights, flatten_weights, unflatten_weights, params_per_wire
from qml_ga.optimizers.ga import run_ga
from qml_ga.optimizers.classical import train_classical
from qml_ga.metrics.classification import all_metrics

def _predict_logits(circuit, W, b, X):
    # QNode agora retorna só expval; somamos b aqui
    return np.array([circuit(W, x.astype(float).ravel()) + b for x in X], dtype=float)

def _train_one_fold(cfg: Dict[str, Any], X_tr, y_tr, X_va, y_va):
    n_qubits = int(cfg["device"]["wires"])
    depth = int(cfg["ansatz"]["params"]["depth"])
    ansatz_type = cfg["ansatz"]["type"]
    fm_type = cfg["feature_map"]["type"]

    _, circuit = build_vqc(ansatz_type, depth, n_qubits, feature_map=fm_type, shots=cfg["device"].get("shots"))
    P = params_per_wire(ansatz_type)

    # init parâmetros
    W0 = init_weights(ansatz_type, n_qubits, depth, scale=0.1, seed=42)
    b0 = 0.0

    opt_type = cfg["optimizer"]["type"].lower()

    if opt_type == "ga":
        ga = cfg["optimizer"]["ga"]
        num_genes = depth * n_qubits * P + 1  # W flatten + bias

        def f_train(sol_vec):
            w, b = sol_vec[:-1], sol_vec[-1]
            W = unflatten_weights(w, ansatz_type, n_qubits, depth)
            logits = _predict_logits(circuit, W, b, X_tr)
            y_pred = (logits >= 0).astype(float) * 2 - 1
            return float((y_tr == y_pred).mean())  # fitness = acc treino

        sol, fit, _, _ = run_ga(
            eval_solution_fn=f_train,
            num_genes=num_genes,
            num_generations=int(ga["num_generations"]),
            sol_per_pop=int(ga["population_size"]),
            num_parents_mating=max(2, int(ga["population_size"]) // 2),
            keep_parents=int(ga.get("elitism", 2)),
            parent_selection_type=ga.get("selection_type", "tournament"),
            crossover_type=ga.get("crossover_type", "single_point"),
            mutation_type=ga.get("mutation_type", "random"),
            mutation_percent_genes=int(ga.get("mutation_percent_genes", 10)),
            init_range_low=float(ga.get("init_range_low", -3.14)),
            init_range_high=float(ga.get("init_range_high", 3.14)),
            random_seed=42,
        )
        w, b = sol[:-1], sol[-1]
        W = unflatten_weights(w, ansatz_type, n_qubits, depth)
        logits = _predict_logits(circuit, W, b, X_va)
        return all_metrics(y_va, logits), W, float(b)

    else:
        cls = cfg["optimizer"]["classical"]
        Wf, bf, _ = train_classical(
            circuit, qnp.array(W0), b0,
            X_train=X_tr, y_train=y_tr,
            optimizer_type=opt_type, lr=float(cls["lr"]),
            epochs=int(cls["epochs"]), batch_size=int(cls["batch_size"]),
        )
        logits = _predict_logits(circuit, Wf, bf, X_va)
        return all_metrics(y_va, logits), Wf, float(bf)

def run_kfold_experiment(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:
    ensure_dirs(run_dir)

    dm = DataModule(cfg["dataset"])
    X, y, feats = dm.load_all()

    fm_type = cfg["feature_map"]["type"]
    wires = int(cfg["device"]["wires"])
    DataModule.ensure_wires_compatible(fm_type, len(feats), wires)

    k = int(cfg.get("k_folds", 5))
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    folds: List[Dict[str, Any]] = []
    for i, (idx_tr, idx_va) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_va], y[idx_va]
        metrics, Wf, bf = _train_one_fold(cfg, X_tr, y_tr, X_va, y_va)
        m = {"fold": i, **metrics}
        folds.append(m)

    mean_acc = float(np.mean([f["accuracy"] for f in folds]))
    std_acc = float(np.std([f["accuracy"] for f in folds]))
    summary = {
        "objective_name": "accuracy",
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "folds": folds,
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
