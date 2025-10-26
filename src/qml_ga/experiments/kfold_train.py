# src/qml_ga/experiments/kfold_train.py
from typing import Dict, Any, List, Optional, Tuple
import os, json, time
import numpy as np
import pandas as pd
import pennylane.numpy as qnp
from sklearn.model_selection import StratifiedKFold

from qml_ga.utils.io import ensure_dirs, write_yaml
from qml_ga.utils.logger import make_text_logger, append_csv_row
from qml_ga.data.datamodule import DataModule
from qml_ga.circuits.vqc import build_vqc
from qml_ga.ansatz.base import init_weights, unflatten_weights, params_per_wire
from qml_ga.optimizers.ga import run_ga
from qml_ga.optimizers.classical import train_classical
from qml_ga.metrics.classification import all_metrics

def _predict_logits(circuit, W, b, X):
    return np.array([circuit(W, x.astype(float).ravel()) + b for x in X], dtype=float)

def _train_one_fold_get_params(
    cfg: Dict[str, Any],
    X_tr, y_tr, X_va=None, y_va=None,
    progress_logger=None,
    train_progress_csv: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Treina e retorna (W, b). Faz logging/CSV durante o treinamento."""
    n_qubits = int(cfg["device"]["wires"])
    depth = int(cfg["ansatz"]["params"]["depth"])
    ansatz_type = cfg["ansatz"]["type"]
    fm_type = cfg["feature_map"]["type"]

    _, circuit = build_vqc(ansatz_type, depth, n_qubits, feature_map=fm_type, shots=cfg["device"].get("shots"))
    P = params_per_wire(ansatz_type)

    # init
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
            return float((y_tr == y_pred).mean())  # fitness: acc treino

        def metrics_from_solution(sol_vec):
            w, b = sol_vec[:-1], sol_vec[-1]
            W = unflatten_weights(w, ansatz_type, n_qubits, depth)
            m_tr = all_metrics(y_tr, _predict_logits(circuit, W, b, X_tr))
            out = { "train_acc": m_tr["accuracy"], "train_f1": m_tr["f1"] }
            if X_va is not None and y_va is not None:
                m_va = all_metrics(y_va, _predict_logits(circuit, W, b, X_va))
                out.update({ "val_acc": m_va["accuracy"], "val_f1": m_va["f1"] })
            return out

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
            progress_logger=progress_logger,
            progress_csv_path=train_progress_csv,
            log_every=1,
            metrics_from_solution=metrics_from_solution,
        )
        w, b = sol[:-1], sol[-1]
        W = unflatten_weights(w, ansatz_type, n_qubits, depth)
        return W, float(b)

    # clássicos: log por batch/epoch + métricas opcionais de validação
    cls = cfg["optimizer"]["classical"]
    Wf, bf, _ = train_classical(
        circuit, qnp.array(W0), b0,
        X_train=X_tr, y_train=y_tr,
        optimizer_type=opt_type, lr=float(cls["lr"]),
        epochs=int(cls["epochs"]), batch_size=int(cls["batch_size"]),
        progress_logger=progress_logger,
        progress_csv_path=train_progress_csv,
        log_batch_every=max(1, int(np.ceil(len(X_tr) / int(cls["batch_size"])) // 4)),  # 4 logs por época
        eval_every=1,
        X_val=X_va, y_val=y_va,
    )
    return np.array(Wf), float(bf)

def run_kfold_experiment(
    cfg: Dict[str, Any],
    run_dir: str,
    report_dir: Optional[str] = None,
    progress: bool = False,
    progress_prefix: str = "",
) -> Dict[str, Any]:
    """Executa K-Fold com logging detalhado de treinamento e salvamento incremental."""
    ensure_dirs(run_dir)
    if report_dir:
        ensure_dirs(report_dir)

    dm = DataModule(cfg["dataset"])
    X, y, feats = dm.load_all()

    fm_type = cfg["feature_map"]["type"]
    wires = int(cfg["device"]["wires"])
    DataModule.ensure_wires_compatible(fm_type, len(feats), wires)

    k = int(cfg.get("k_folds", 5))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # arquivos de saída por configuração
    folds_csv = os.path.join(report_dir or run_dir, "folds.csv")
    train_csv = os.path.join(report_dir or run_dir, "train_progress.csv")
    train_log = os.path.join(report_dir or run_dir, "train.log")
    status_json = os.path.join(report_dir or run_dir, "status.json")
    summary_json = os.path.join(report_dir or run_dir, "summary.json")
    cfg_yaml = os.path.join(report_dir or run_dir, "config.yaml")

    # logger de treinamento
    logger = make_text_logger(train_log, prefix=f"{progress_prefix}  ")

    # snapshot de config
    try:
        write_yaml(cfg, cfg_yaml)
    except Exception:
        pass

    # status inicial
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    status = {"started_at": started_at, "k_folds": k, "last_fold_completed": 0, "done": False, "failed": False}
    with open(status_json, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    if progress:
        print(
            f"{progress_prefix} dataset={cfg['dataset']['name']} | fm={fm_type} | "
            f"ansatz={cfg['ansatz']['type']} d={cfg['ansatz']['params']['depth']} | "
            f"opt={cfg['optimizer']['type']} | wires={wires}",
            flush=True,
        )

    folds: List[Dict[str, Any]] = []
    t0_all = time.time()

    # constrói circuito uma vez para predições
    _, circuit = build_vqc(cfg["ansatz"]["type"], int(cfg["ansatz"]["params"]["depth"]), wires,
                           feature_map=fm_type, shots=cfg["device"].get("shots"))

    for i, (idx_tr, idx_va) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_va], y[idx_va]

        if progress:
            print(f"{progress_prefix}  [fold {i}/{k}] training...", flush=True)
        logger(f"[fold {i}/{k}] start")

        # treina (com logs/CSV internos)
        Wf, bf = _train_one_fold_get_params(
            cfg, X_tr, y_tr, X_va, y_va,
            progress_logger=logger,
            train_progress_csv=train_csv,
        )

        # métricas finais do fold
        logits_tr = _predict_logits(circuit, Wf, bf, X_tr)
        m_tr = all_metrics(y_tr, logits_tr)
        logits_va = _predict_logits(circuit, Wf, bf, X_va)
        m_va = all_metrics(y_va, logits_va)
        dt = time.time() - t0

        row = {
            "fold": i,
            "train_accuracy": m_tr["accuracy"],
            "train_precision": m_tr["precision"],
            "train_recall": m_tr["recall"],
            "train_f1": m_tr["f1"],
            "val_accuracy": m_va["accuracy"],
            "val_precision": m_va["precision"],
            "val_recall": m_va["recall"],
            "val_f1": m_va["f1"],
            "seconds": float(dt),
        }
        folds.append(row)
        append_csv_row(folds_csv, row)

        status["last_fold_completed"] = i
        status["last_fold_seconds"] = float(dt)
        with open(status_json, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        logger(f"[fold {i}/{k}] "
               f"train acc={m_tr['accuracy']:.4f} f1={m_tr['f1']:.4f} | "
               f"val acc={m_va['accuracy']:.4f} f1={m_va['f1']:.4f} ({dt:.1f}s)")

    # resumo (médias nas dobras) — validação
    mean_val = {
        "accuracy": float(np.mean([f["val_accuracy"] for f in folds])),
        "precision": float(np.mean([f["val_precision"] for f in folds])),
        "recall": float(np.mean([f["val_recall"] for f in folds])),
        "f1": float(np.mean([f["val_f1"] for f in folds])),
    }
    std_val = {
        "accuracy": float(np.std([f["val_accuracy"] for f in folds])),
        "precision": float(np.std([f["val_precision"] for f in folds])),
        "recall": float(np.std([f["val_recall"] for f in folds])),
        "f1": float(np.std([f["val_f1"] for f in folds])),
    }
    mean_train = {
        "accuracy": float(np.mean([f["train_accuracy"] for f in folds])),
        "precision": float(np.mean([f["train_precision"] for f in folds])),
        "recall": float(np.mean([f["train_recall"] for f in folds])),
        "f1": float(np.mean([f["train_f1"] for f in folds])),
    }

    total_s = time.time() - t0_all
    summary = {
        "objective_name": "accuracy",
        "seconds_total": float(total_s),
        "val_mean": mean_val,
        "val_std": std_val,
        "train_mean": mean_train,
        "folds": folds,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    status["done"] = True
    with open(status_json, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    if progress:
        print(
            f"{progress_prefix}  [done] "
            f"val(mean): acc={mean_val['accuracy']:.4f} p={mean_val['precision']:.4f} r={mean_val['recall']:.4f} f1={mean_val['f1']:.4f} "
            f"total={total_s/60:.1f}min",
            flush=True,
        )

    return summary
