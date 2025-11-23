from typing import Callable, Dict, Optional, Tuple, Any
import os, time, json
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp

from qml_ga.metrics.classification import all_metrics
from qml_ga.utils.logger import append_csv_row

OptimType = qml.optimize.AdamOptimizer
NesterovType = qml.optimize.NesterovMomentumOptimizer

def _make_optimizer(opt_name: str, lr: float):
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return qml.optimize.AdamOptimizer(stepsize=lr)
    if opt_name in ("nesterov", "nesterov_momentum"):
        return qml.optimize.NesterovMomentumOptimizer(stepsize=lr)
    raise ValueError(f"Unsupported optimizer {opt_name}")

def _batch_idx(n: int, batch_size: int, epoch: int):
    # baralha por época de forma determinística
    rng = np.random.default_rng(42 + epoch)
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i+batch_size]

def train_classical(
    circuit: Callable,
    W0: qnp.ndarray,
    b0: float,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer_type: str,
    lr: float,
    epochs: int,
    batch_size: int,
    progress_logger: Optional[Callable[[str], None]] = None,
    train_progress_csv: Optional[str] = None,
    eval_every: int = 1,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    objective: str = "accuracy",
    mid_checkpoint_epoch: int = 100,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[qnp.ndarray, float, Dict[str, Any]]:
    """
    Treina com Adam ou Nesterov e retorna:
      W_final, b_final, extras
    extras inclui mid_checkpoint com métricas até a época mid_checkpoint_epoch
    e caminho do arquivo salvo.

    O loss é MSE nos logits contra rótulos em {-1,+1}.
    """

    def predict_logits(W, b, X):
        return qnp.array([circuit(W, x) + b for x in X])

    def loss_mse(W, b, Xe, ye):
        preds = predict_logits(W, b, Xe)
        return qnp.mean((ye - preds) ** 2)

    opt = _make_optimizer(optimizer_type, lr)

    W = W0.copy()
    b = qnp.array(b0)

    best_mid = None
    mid_saved = False
    mid_epoch = int(mid_checkpoint_epoch)

    n = X_train.shape[0]
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # minibatches
        for mb in _batch_idx(n, max(1, batch_size), epoch):
            Xe = X_train[mb]
            ye = y_train[mb]

            # faz um passo de otimização conjunto em (W,b)
            def cost(params_W, params_b):
                return loss_mse(params_W, params_b, Xe, ye)

            W, b = opt.step(cost, W, b)

        # avaliação por época
        if eval_every and (epoch % eval_every == 0):
            # train
            logits_tr = predict_logits(W, b, X_train)
            m_tr = all_metrics(y_train, logits_tr)

            # val
            m_va = {}
            if X_val is not None and y_val is not None:
                logits_va = predict_logits(W, b, X_val)
                m_va = all_metrics(y_val, logits_va)

            # logging
            if progress_logger:
                if m_va:
                    progress_logger(
                        f"epoch {epoch}/{epochs} train acc={m_tr['accuracy']:.4f} f1={m_tr['f1']:.4f} "
                        f"| val acc={m_va['accuracy']:.4f} f1={m_va['f1']:.4f}"
                    )
                else:
                    progress_logger(
                        f"epoch {epoch}/{epochs} train acc={m_tr['accuracy']:.4f} f1={m_tr['f1']:.4f}"
                    )

            if train_progress_csv is not None:
                append_csv_row(train_progress_csv, {
                    "epoch": epoch,
                    "train_accuracy": float(m_tr["accuracy"]),
                    "train_precision": float(m_tr["precision"]),
                    "train_recall": float(m_tr["recall"]),
                    "train_f1": float(m_tr["f1"]),
                    "val_accuracy": float(m_va.get("accuracy", np.nan)),
                    "val_precision": float(m_va.get("precision", np.nan)),
                    "val_recall": float(m_va.get("recall", np.nan)),
                    "val_f1": float(m_va.get("f1", np.nan)),
                })

            # guarda o melhor até a época 100
            if epoch <= mid_epoch and m_va:
                key = "accuracy" if objective == "accuracy" else "f1"
                score = m_va.get(key, 0.0)
                if best_mid is None or score > best_mid["score"]:
                    best_mid = {
                        "epoch": epoch,
                        "score": float(score),
                        "metrics": {k: float(v) for k, v in m_va.items()},
                        "train_metrics": {k: float(v) for k, v in m_tr.items()},
                        "W": np.asarray(W).tolist(),
                        "b": float(b),
                    }

            if (not mid_saved) and epoch >= mid_epoch:
                # salva snapshot com o melhor até a época 100
                mid_saved = True
                if checkpoint_dir is not None:
                    path = os.path.join(checkpoint_dir, "checkpoint_epoch100.json")
                    payload = best_mid if best_mid is not None else {
                        "epoch": mid_epoch,
                        "score": None,
                        "metrics": {},
                        "train_metrics": {},
                        "W": np.asarray(W).tolist(),
                        "b": float(b),
                    }
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)

    # fim do treino
    extras = {
        "mid_checkpoint": best_mid,
        "objective": objective,
        "seconds_train": float(time.time() - t0),
    }
    return W, float(b), extras
