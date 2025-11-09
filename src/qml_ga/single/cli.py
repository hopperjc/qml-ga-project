import os, json, argparse, time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from qml_ga.utils.io import load_yaml, write_yaml, ensure_dirs, setup_cwd_to_repo_root, make_run_dir
from qml_ga.utils.logger import make_text_logger, append_csv_row
from qml_ga.data.datamodule import DataModule
from qml_ga.circuits.vqc import build_vqc
from qml_ga.metrics.classification import all_metrics
from qml_ga.experiments.kfold_train import _train_one_fold_get_params


def _derive_device(dataset_block: Dict, feature_map_block: Dict, wires_override: Optional[int], device_name="default.qubit", shots=None) -> Dict:
    dm = DataModule(dataset_block)
    X, y, feats = dm.load_all()
    fm_type = feature_map_block["type"]
    if wires_override is None:
        wires = DataModule.required_wires(fm_type, len(feats))
    else:
        wires = int(wires_override)
    DataModule.ensure_wires_compatible(fm_type, len(feats), wires)
    return {"device": {"name": device_name, "wires": wires, "shots": shots}}, (X, y, feats)


def _tag(cfg: Dict[str, Any], params: Dict[str, Any]) -> str:
    ds = cfg["dataset"]["name"]
    fm = cfg["feature_map"]["type"]
    at = cfg["ansatz"]["type"]
    d = cfg["ansatz"]["params"].get("depth", "")
    opt = cfg["optimizer"]["type"]
    w = cfg["device"]["wires"]
    return f"{ds}__{fm}__{at}_d{d}__{opt}__w{w}"


def main(
    dataset_yaml: str,
    feature_map_yaml: str,
    ansatz_yaml: str,
    optimizer: str,
    reports_dir: str = "reports",
    runs_dir: str = "runs",
    val_size: float = 0.2,
    device: str = "default.qubit",
    wires: Optional[int] = None,
    shots: Optional[int] = None,
    # clássicos
    lr: float = 0.05,
    epochs: int = 50,
    batch_size: int = 16,
    # GA
    population_size: int = 50,
    num_generations: int = 50,
    selection_type: str = "tournament",
    crossover_type: str = "single_point",
    mutation_type: str = "random",
    mutation_percent_genes: int = 10,
    elitism: int = 2,
    init_range_low: float = -3.14,
    init_range_high: float = 3.14,
    # logging
    log_every: int = 1,
):
    """
    Executa UM ÚNICO RUN (holdout estratificado) para checagem rápida.
    Salva: train.log, train_progress.csv, folds.csv, summary.json, status.json e config.yaml em reports/<TAG>__SINGLE/.
    """
    setup_cwd_to_repo_root()
    ensure_dirs(reports_dir, runs_dir)

    # carrega YAMLs
    ds_block = load_yaml(dataset_yaml)["dataset"]
    fm_block = load_yaml(feature_map_yaml)["feature_map"]
    an_block = load_yaml(ansatz_yaml)["ansatz"]

    # device + dados
    dev_block, (X, y, feats) = _derive_device(ds_block, fm_block, wires_override=wires, device_name=device, shots=shots)

    # configura otimizador
    optimizer = optimizer.lower().strip()
    if optimizer in ("adam", "nesterov"):
        opt_block = {"type": optimizer, "classical": {"lr": lr, "epochs": epochs, "batch_size": batch_size}}
        params_for_tag = {"lr": lr, "epochs": epochs, "batch_size": batch_size}
    elif optimizer == "ga":
        opt_block = {
            "type": "ga",
            "ga": {
                "population_size": population_size,
                "num_generations": num_generations,
                "selection_type": selection_type,
                "crossover_type": crossover_type,
                "mutation_type": mutation_type,
                "mutation_percent_genes": mutation_percent_genes,
                "elitism": elitism,
                "init_range_low": init_range_low,
                "init_range_high": init_range_high,
            },
        }
        params_for_tag = opt_block["ga"]
    else:
        raise ValueError(f"optimizer inválido: {optimizer}")

    # cfg consolidada (compatível com as funções existentes)
    cfg = {
        "dataset": ds_block,
        "feature_map": fm_block,
        "ansatz": an_block,
        "device": dev_block["device"],
        "optimizer": opt_block,
        "seed": 42,
        "k_folds": 2,  # só para manter compatibilidade caso alguma função use; NÃO usado aqui.
        "output": {"base_dir": runs_dir, "save_specs": False},
    }

    tag = _tag(cfg, params_for_tag)
    tag = f"{tag}__SINGLE"
    rep_dir = os.path.join(reports_dir, tag)
    ensure_dirs(rep_dir)
    run_dir = make_run_dir(runs_dir)

    # paths de saída
    train_log = os.path.join(rep_dir, "train.log")
    train_csv = os.path.join(rep_dir, "train_progress.csv")
    folds_csv = os.path.join(rep_dir, "folds.csv")
    status_json = os.path.join(rep_dir, "status.json")
    summary_json = os.path.join(rep_dir, "summary.json")
    cfg_yaml = os.path.join(rep_dir, "config.yaml")
    write_yaml(cfg, cfg_yaml)

    logger = make_text_logger(train_log, prefix=f"[{tag}] ")

    # split holdout estratificado
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    (idx_tr, idx_va), = sss.split(X, y)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]

    # status inicial
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    status = {"started_at": started_at, "split": "holdout", "val_size": val_size, "done": False, "failed": False}
    with open(status_json, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    # circuito (para predições após treino)
    _, circuit = build_vqc(cfg["ansatz"]["type"], int(cfg["ansatz"]["params"]["depth"]), int(cfg["device"]["wires"]),
                           feature_map=cfg["feature_map"]["type"], shots=cfg["device"].get("shots"))

    # treino
    logger("START")
    t0 = time.time()
    try:
        Wf, bf = _train_one_fold_get_params(
            cfg,
            X_tr, y_tr,
            X_va=X_va, y_va=y_va,
            progress_logger=logger,
            train_progress_csv=train_csv,
        )
    except Exception as e:
        logger(f"FAIL: {e}")
        status["failed"] = True
        with open(status_json, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        raise

    # métricas
    def _predict_logits(circuit, W, b, X):
        return np.array([circuit(W, x.astype(float).ravel()) + b for x in X], dtype=float)

    logits_tr = _predict_logits(circuit, Wf, bf, X_tr)
    m_tr = all_metrics(y_tr, logits_tr)
    logits_va = _predict_logits(circuit, Wf, bf, X_va)
    m_va = all_metrics(y_va, logits_va)
    dt = time.time() - t0

    # salva fold único em CSV
    row = {
        "fold": 1,
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
    append_csv_row(folds_csv, row)

    summary = {
        "objective_name": "accuracy",
        "seconds_total": float(dt),
        "val_mean": {
            "accuracy": float(m_va["accuracy"]),
            "precision": float(m_va["precision"]),
            "recall": float(m_va["recall"]),
            "f1": float(m_va["f1"]),
        },
        "train_mean": {
            "accuracy": float(m_tr["accuracy"]),
            "precision": float(m_tr["precision"]),
            "recall": float(m_tr["recall"]),
            "f1": float(m_tr["f1"]),
        },
        "folds": [row],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    status["done"] = True
    with open(status_json, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    logger(f"DONE val acc={m_va['accuracy']:.4f} p={m_va['precision']:.4f} r={m_va['recall']:.4f} f1={m_va['f1']:.4f} ({dt:.1f}s)")
    print(f"[SINGLE] {tag} OK  val_acc={m_va['accuracy']:.4f}")


def entrypoint():
    ap = argparse.ArgumentParser(description="Single-run (holdout) para checar uma configuração específica (clássicos ou GA).")
    ap.add_argument("--dataset_yaml", required=True, help="Ex.: configs/datasets/banknote_authentication.yaml")
    ap.add_argument("--feature_map_yaml", required=True, help="Ex.: configs/feature_maps/amplitude.yaml")
    ap.add_argument("--ansatz_yaml", required=True, help="Ex.: configs/ansatz/ansatz_1_d15.yaml")
    ap.add_argument("--optimizer", required=True, choices=["adam","nesterov","ga"])

    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--device", default="default.qubit")
    ap.add_argument("--wires", type=int, default=None)
    ap.add_argument("--shots", type=int, default=None)

    # clássicos
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)

    # GA
    ap.add_argument("--population_size", type=int, default=50)
    ap.add_argument("--num_generations", type=int, default=50)
    ap.add_argument("--selection_type", default="tournament")
    ap.add_argument("--crossover_type", default="single_point")
    ap.add_argument("--mutation_type", default="random")
    ap.add_argument("--mutation_percent_genes", type=int, default=10)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--init_range_low", type=float, default=-3.14)
    ap.add_argument("--init_range_high", type=float, default=3.14)

    ap.add_argument("--log_every", type=int, default=1)

    args = ap.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    entrypoint()
