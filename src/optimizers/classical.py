from typing import Tuple, List, Callable, Optional
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from .base import BaseOptimizer

def _binary_ce_from_probs(p1: pnp.ndarray, y: pnp.ndarray) -> pnp.float64:
    eps = 1e-8
    p1 = pnp.clip(p1, eps, 1 - eps)
    y = y.astype(pnp.float64)
    return -pnp.mean(y * pnp.log(p1) + (1 - y) * pnp.log(1 - p1))

def make_cost(vqc, X: np.ndarray, y: np.ndarray) -> Callable[[pnp.ndarray], pnp.float64]:
    circ = vqc.qnode()
    Xp = pnp.array(X, dtype=pnp.float64)
    yp = pnp.array(y, dtype=pnp.float64)
    def cost(weights: pnp.ndarray) -> pnp.float64:
        z = pnp.array([circ(weights, x) for x in Xp])  # [-1, 1]
        p1 = 0.5 * (1 - z)
        return _binary_ce_from_probs(p1, yp)
    return cost

class AdamOptimizer(BaseOptimizer):
    def __init__(self, vqc, ansatz, X_train, y_train, lr=0.05, beta1=0.9, beta2=0.999,
                 eps=1e-8, epochs=200, batch_size=None, l2=0.0, seed=42):
        self.vqc, self.ansatz = vqc, ansatz
        self.X, self.y = X_train, y_train
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.epochs, self.batch_size, self.l2 = epochs, batch_size, l2
        self.rng = np.random.default_rng(seed)
        self.n_params = ansatz.num_params(vqc.dev.num_wires)

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        w = pnp.array(self.rng.uniform(-np.pi, np.pi, size=self.n_params), dtype=pnp.float64)
        m = pnp.zeros_like(w); v = pnp.zeros_like(w)
        full_cost = make_cost(self.vqc, self.X, self.y); cost_grad = qml.grad(full_cost)
        history, best_w, best_loss = [], pnp.copy(w), pnp.inf

        for t in range(1, self.epochs + 1):
            if self.batch_size and self.batch_size < len(self.y):
                idx = self.rng.choice(len(self.y), size=self.batch_size, replace=False)
                batch_cost = make_cost(self.vqc, self.X[idx], self.y[idx])
                g = cost_grad(w); loss = batch_cost(w)
            else:
                g = cost_grad(w); loss = full_cost(w)

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * pnp.sum(w * w)
                g = g + self.l2 * w

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** t); v_hat = v / (1 - self.beta2 ** t)
            w = w - self.lr * m_hat / (pnp.sqrt(v_hat) + self.eps)

            wl = float(loss); history.append(wl)
            if wl < best_loss: best_loss, best_w = wl, pnp.copy(w)

        return np.array(best_w, dtype=float), float(best_loss), [float(h) for h in history]

class NesterovOptimizer(BaseOptimizer):
    def __init__(self, vqc, ansatz, X_train, y_train, lr=0.05, momentum=0.9,
                 epochs=200, batch_size=None, l2=0.0, seed=42):
        self.vqc, self.ansatz = vqc, ansatz
        self.X, self.y = X_train, y_train
        self.lr, self.mu, self.epochs, self.batch_size, self.l2 = lr, momentum, epochs, batch_size, l2
        self.rng = np.random.default_rng(seed)
        self.n_params = ansatz.num_params(vqc.dev.num_wires)

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        w = pnp.array(self.rng.uniform(-np.pi, np.pi, size=self.n_params), dtype=pnp.float64)
        v = pnp.zeros_like(w)
        full_cost = make_cost(self.vqc, self.X, self.y); cost_grad = qml.grad(full_cost)
        history, best_w, best_loss = [], pnp.copy(w), pnp.inf

        for _ in range(self.epochs):
            if self.batch_size and self.batch_size < len(self.y):
                idx = self.rng.choice(len(self.y), size=self.batch_size, replace=False)
                batch_cost = make_cost(self.vqc, self.X[idx], self.y[idx])
                g = cost_grad(w + self.mu * v); loss = batch_cost(w)
            else:
                g = cost_grad(w + self.mu * v); loss = full_cost(w)

            if self.l2 > 0.0:
                loss = loss + 0.5 * self.l2 * pnp.sum(w * w)
                g = g + self.l2 * w

            v = self.mu * v - self.lr * g
            w = w + v

            wl = float(loss); history.append(wl)
            if wl < best_loss: best_loss, best_w = wl, pnp.copy(w)

        return np.array(best_w, dtype=float), float(best_loss), [float(h) for h in history]
