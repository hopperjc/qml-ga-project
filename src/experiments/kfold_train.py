import os, csv, json
from typing import Dict, Tuple, List
import numpy as np
import pennylane as qml

from data.datamodule import DataModule
from feature_maps.amplitude_encoding import AmplitudeEncoding
from feature_maps.zz_feature_map import ZZFeatureMap
from ansatz import REGISTRY as ANSATZ_REGISTRY
from circuits.vqc import VQC
from optimizers.ga import GAOptimizer
from optimizers.classical import AdamOptimizer, NesterovOptimizer
from metrics.classification import compute_metrics
from utils.io import write_json
from utils.plotting import plot_fitness

def build_device(device_cfg: Dict):
    name = device_cfg.get("name", "default.qubit")
    wires = int(device_cfg.get("wires", 4))
    shots = device_cfg.get("shots", None)
    return qml.device(name, wires=wires, shots=shots)

def build_feature_map(cfg: Dict):
    ftype = cfg["feature_map"]["type"].lower()
    params = cfg["feature_map"].get("params", {}) or {}
    if ftype == "amplitude": return AmplitudeEncoding(**params)
    if ftype == "zz":        return ZZFeatureMap(**params)
    raise ValueError(f"feature_map não suportado: {ftype}")

def build_ansatz(cfg: Dict):
    atype = cfg["ansatz"]["type"]
    params = cfg["ansatz"].get("params", {}) or {}
    if atype not in ANSATZ_REGISTRY:
        raise ValueError(f"ansatz não suportado: {atype}. Opções: {list(ANSATZ_REGISTRY.keys())}")
    return ANSATZ_REGISTRY[atype](**params)

def _apply_smote_if_needed(dataset_cfg: Dict, Xtr: np.ndarray, ytr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    balance = dataset_cfg.get("balance", {}) or {}
    method = (balance.get("method") or "").lower()
    if method != "smote": return Xtr, ytr
    k_neighbors = int(balance.get("k_neighbors", 5))
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:
        raise RuntimeError("SMOTE requisitado, mas imbalanced-learn não está instalado.") from e
    skw = dataset_cfg.get("kfold", {}) or {}
    sm = SMOTE(k_neighbors=k_neighbors, random_state=int(skw.get("seed", 42)))
    return sm.fit_resample(Xtr, ytr)

def _build_optimizer(opt_cfg: Dict, vqc: VQC, ansatz, Xtr: np.ndarray, ytr: np.ndarray):
    otype = opt_cfg["type"].lower()
    if otype == "ga":
        return GAOptimizer(vqc, ansatz, opt_cfg.get("ga", {}) or {}, Xtr, ytr)
    if otype == "adam":
        c = opt_cfg.get("classical", {}) or {}
        return AdamOptimizer(vqc, ansatz, Xtr, ytr, lr=c.get("lr",0.05), beta1=c.get("beta1",0.9),
                             beta2=c.get("beta2",0.999), eps=c.get("eps",1e-8), epochs=c.get("epochs",200),
                             batch_size=c.get("batch_size",None), l2=c.get("l2",0.0), seed=c.get("seed",42))
    if otype in ("nesterov","momentum"):
        c = opt_cfg.get("classical", {}) or {}
        return NesterovOptimizer(vqc, ansatz, Xtr, ytr, lr=c.get("lr",0.05), momentum=c.get("momentum",0.9),
                                 epochs=c.get("epochs",200), batch_size=c.get("batch_size",None),
                                 l2=c.get("l2",0.0), seed=c.get("seed",42))
    raise ValueError(f"optimizer.type não suportado: {otype}")

def _objective_name(opt_cfg: Dict) -> str:
    return "fitness" if opt_cfg["type"].lower() == "ga" else "loss"

def _save_fold_artifacts(out_dir: str, fold_id: int, best_w: np.ndarray, history: List[float], objective_name: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"best_params_fold_{fold_id:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(best_w.tolist(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"history_fold_{fold_id:02d}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([objective_name])
        for v in history: w.writerow([float(v)])

def run_kfold_experiment(cfg: Dict, out_dir: str) -> Dict:
    dm = DataModule(cfg["dataset"])
    X, y, feats = dm.load_all()

    fm_type = cfg["feature_map"]["type"]
    dev_wires = int(cfg["device"]["wires"])
    DataModule.ensure_wires_compatible(fm_type, len(feats), dev_wires)

    device = build_device(cfg["device"])
    feature_map = build_feature_map(cfg)
    ansatz = build_ansatz(cfg)
    opt_cfg = cfg["optimizer"]; obj_name = _objective_name(opt_cfg)

    all_metrics, all_histories = [], []
    save_specs = bool(cfg.get("output", {}).get("save_specs", False)); specs_saved = False

    fold_id = 0
    for tr, te in dm.folds(X, y):
        fold_id += 1
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        Xtr, ytr = _apply_smote_if_needed(cfg["dataset"], Xtr, ytr)

        vqc = VQC(device, feature_map, ansatz)
        opt = _build_optimizer(opt_cfg, vqc, ansatz, Xtr, ytr)

        best_w, train_obj_best, history = opt.run()

        y_pred = vqc.predict(best_w, Xte)
        m = compute_metrics(yte, y_pred)
        m["fold"] = fold_id
        m["train_objective"] = float(train_obj_best)
        m["objective_name"] = obj_name
        all_metrics.append(m)
        all_histories.append(history)

        _save_fold_artifacts(out_dir, fold_id, best_w, history, obj_name)

        if save_specs and not specs_saved:
            try:
                from circuits.inspect import specs_snapshot
                specs_snapshot(vqc.qnode(), best_w, Xtr[0], os.path.join(out_dir, "specs.json"))
                specs_saved = True
            except Exception:
                pass

    accs = [m["accuracy"] for m in all_metrics]
    summary = {
        "folds": all_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "objective_name": obj_name,
    }
    write_json(summary, os.path.join(out_dir, "metrics.json"))
    try: plot_fitness(all_histories[-1], os.path.join(out_dir, "fitness_curve.png"))
    except Exception: pass
    return summary
