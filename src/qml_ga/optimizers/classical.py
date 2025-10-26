# src/qml_ga/optimizers/classical.py
from typing import Tuple, Callable, Optional
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
from sklearn.utils import shuffle
from qml_ga.metrics.classification import all_metrics
from qml_ga.utils.logger import append_csv_row

ProgressLogger = Optional[Callable[[str], None]]

def _make_loss(circuit, X, y):
    y = qnp.array(y, dtype=float)
    def loss(W, b):
        preds = qnp.array([circuit(W, x) + b for x in X])
        return qnp.mean((preds - y) ** 2)
    return loss

def train_classical(
    circuit,
    W0: qnp.ndarray,
    b0: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer_type: str = "adam",
    lr: float = 0.05,
    epochs: int = 50,
    batch_size: int = 16,
    # logging/progresso
    progress_logger: ProgressLogger = None,
    progress_csv_path: Optional[str] = None,
    log_batch_every: int = 0,      # 0 = não logar por batch; >0 = logar a cada N batches
    eval_every: int = 1,           # avalia métricas no fim de cada N épocas
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[qnp.ndarray, float, float]:
    """
    Treino MSE com Adam/Nesterov (PennyLane). Retorna (W, b, final_loss).
    Faz logging por batch e por epoch, gravando CSV incremental.
    """
    optimizer_type = (optimizer_type or "adam").lower()
    if optimizer_type == "adam":
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_type == "nesterov":
        try:
            opt = qml.NesterovMomentumOptimizer(stepsize=lr, momentum=0.9)
        except AttributeError:
            opt = qml.GradientDescentOptimizer(stepsize=lr)
    else:
        raise ValueError(f"Otimizador desconhecido: {optimizer_type}")

    W = qnp.array(W0, requires_grad=True)
    b = qnp.array(b0, requires_grad=True)

    n = len(X_train)
    y_train = qnp.array(y_train, dtype=float)
    loss_fn = _make_loss(circuit, X_train, y_train)

    total_batches = int(np.ceil(n / batch_size))
    for ep in range(epochs):
        X_sh, y_sh = shuffle(X_train, y_train, random_state=ep)
        epoch_loss_vals = []

        for bi, s in enumerate(range(0, n, batch_size), start=1):
            Xe = qnp.array(X_sh[s:s + batch_size], requires_grad=False)
            ye = qnp.array(y_sh[s:s + batch_size], requires_grad=False)

            def step_loss(W_, b_):
                preds = qnp.array([circuit(W_, x) + b_ for x in Xe])
                return qnp.mean((preds - ye) ** 2)

            res = opt.step_and_cost(step_loss, W, b)
            # Compat: (W,b,loss) ou ((W,b),loss)
            if isinstance(res, (list, tuple)) and len(res) == 3:
                W, b, l = res
            else:
                (W, b), l = res

            epoch_loss_vals.append(float(l))

            if log_batch_every and (bi % log_batch_every == 0 or bi == total_batches):
                msg = f"epoch {ep+1}/{epochs} batch {bi}/{total_batches} loss={float(l):.6f}"
                if progress_logger: progress_logger(msg)
                if progress_csv_path:
                    append_csv_row(progress_csv_path, {
                        "phase": "train_batch",
                        "epoch": ep+1,
                        "batch": bi,
                        "batches_per_epoch": total_batches,
                        "loss": float(l),
                    })

        # fim da época: métricas de treino
        preds_tr = qnp.array([circuit(W, x) + b for x in X_train])
        m_tr = all_metrics(np.array(y_train), np.array(preds_tr))
        msg_tail = f"train acc={m_tr['accuracy']:.4f} p={m_tr['precision']:.4f} r={m_tr['recall']:.4f} f1={m_tr['f1']:.4f} mean_loss={np.mean(epoch_loss_vals):.6f}"

        # validação opcional por época
        if (ep + 1) % max(1, eval_every) == 0 and X_val is not None and y_val is not None:
            preds_va = qnp.array([circuit(W, x) + b for x in X_val])
            m_va = all_metrics(np.array(y_val), np.array(preds_va))
            msg = f"epoch {ep+1}/{epochs} {msg_tail} | val acc={m_va['accuracy']:.4f} p={m_va['precision']:.4f} r={m_va['recall']:.4f} f1={m_va['f1']:.4f}"
            if progress_logger: progress_logger(msg)
            if progress_csv_path:
                append_csv_row(progress_csv_path, {
                    "phase": "epoch_end",
                    "epoch": ep+1,
                    "loss": float(np.mean(epoch_loss_vals)),
                    "train_accuracy": m_tr["accuracy"],
                    "train_precision": m_tr["precision"],
                    "train_recall": m_tr["recall"],
                    "train_f1": m_tr["f1"],
                    "val_accuracy": m_va["accuracy"],
                    "val_precision": m_va["precision"],
                    "val_recall": m_va["recall"],
                    "val_f1": m_va["f1"],
                })
        else:
            msg = f"epoch {ep+1}/{epochs} {msg_tail}"
            if progress_logger: progress_logger(msg)
            if progress_csv_path:
                append_csv_row(progress_csv_path, {
                    "phase": "epoch_end",
                    "epoch": ep+1,
                    "loss": float(np.mean(epoch_loss_vals)),
                    "train_accuracy": m_tr["accuracy"],
                    "train_precision": m_tr["precision"],
                    "train_recall": m_tr["recall"],
                    "train_f1": m_tr["f1"],
                })

    final_loss = float(loss_fn(W, b))
    return W, float(b), final_loss
