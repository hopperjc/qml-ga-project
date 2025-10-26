from typing import Dict, Sequence
import pennylane.numpy as qnp

# parâmetros por wire em UMA camada
PARAMS_PER_WIRE: Dict[str, int] = {
    "ansatz_1": 3,
    "ansatz_2": 3,
    "ansatz_3": 5,
    "ansatz_4": 5,
    "ansatz_5": 5,
    "ansatz_6": 3,
}

def params_per_wire(ansatz_type: str) -> int:
    if ansatz_type not in PARAMS_PER_WIRE:
        raise ValueError(f"Ansatz desconhecido: {ansatz_type}")
    return PARAMS_PER_WIRE[ansatz_type]

def init_weights(ansatz_type: str, n_wires: int, depth: int, scale: float = 0.1, seed: int = 42):
    """W shape: (depth, n_wires, P)"""
    qnp.random.seed(seed)
    P = params_per_wire(ansatz_type)
    return qnp.random.uniform(-scale, scale, size=(depth, n_wires, P))

def flatten_weights(W):
    return qnp.ravel(W)

def unflatten_weights(w_flat, ansatz_type: str, n_wires: int, depth: int):
    P = params_per_wire(ansatz_type)
    return qnp.array(w_flat, dtype=float).reshape((depth, n_wires, P))

# ---- dispatcher: aplica depth camadas, cada uma consome W[d] (n_wires, P) ----
from .ansatz_1 import ansatz_1
from .ansatz_2 import ansatz_2
from .ansatz_3 import ansatz_3
from .ansatz_4 import ansatz_4
from .ansatz_5 import ansatz_5
from .ansatz_6 import ansatz_6

def apply_ansatz(ansatz_type: str, W, wires: Sequence[int], depth: int):
    if W.ndim != 3:
        raise ValueError(f"Esperado W com 3 dims (depth, n_wires, P); recebi {W.shape}")
    n_wires = len(wires)
    P = params_per_wire(ansatz_type)
    if W.shape != (depth, n_wires, P):
        raise ValueError(f"Shape de W incompatível: esperado {(depth, n_wires, P)}, recebi {W.shape}")

    for d in range(depth):
        if ansatz_type == "ansatz_1":
            ansatz_1(W[d], wires)
        elif ansatz_type == "ansatz_2":
            ansatz_2(W[d], wires)
        elif ansatz_type == "ansatz_3":
            ansatz_3(W[d], wires)
        elif ansatz_type == "ansatz_4":
            ansatz_4(W[d], wires)
        elif ansatz_type == "ansatz_5":
            ansatz_5(W[d], wires)
        elif ansatz_type == "ansatz_6":
            ansatz_6(W[d], wires)
        else:
            raise ValueError(f"Ansatz desconhecido: {ansatz_type}")
