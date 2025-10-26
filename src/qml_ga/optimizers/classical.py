from typing import Tuple
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
from sklearn.utils import shuffle

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
) -> Tuple[qnp.ndarray, float, float]:
    """
    Treino MSE com Adam/Nesterov (PennyLane). Retorna (W, b, final_loss).
    Compatível com retornos 2-tupla e 3-tupla de step_and_cost.
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

    for ep in range(epochs):
        X_sh, y_sh = shuffle(X_train, y_train, random_state=ep)
        for s in range(0, n, batch_size):
            Xe = qnp.array(X_sh[s:s + batch_size], requires_grad=False)
            ye = qnp.array(y_sh[s:s + batch_size], requires_grad=False)

            def step_loss(W_, b_):
                preds = qnp.array([circuit(W_, x) + b_ for x in Xe])
                return qnp.mean((preds - ye) ** 2)

            res = opt.step_and_cost(step_loss, W, b)
            # PL pode retornar (W,b,loss) OU ((W,b),loss)
            if isinstance(res, (list, tuple)) and len(res) == 3:
                W, b, l = res
            else:
                (W, b), l = res

    final_loss = float(loss_fn(W, b))
    return W, float(b), final_loss
