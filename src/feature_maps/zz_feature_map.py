from typing import Callable, Iterable, List, Sequence, Tuple, Optional
import numpy as np
import pennylane as qml
from .base import IFeatureMap

def _pairs_linear(wires: Sequence[int]) -> List[Tuple[int, int]]:
    return [(wires[i], wires[i+1]) for i in range(len(wires) - 1)]

def _pairs_full(wires: Sequence[int]) -> List[Tuple[int, int]]:
    return [(wires[i], wires[j]) for i in range(len(wires)) for j in range(i+1, len(wires))]

class ZZFeatureMap(IFeatureMap):
    def __init__(
        self,
        reps: int = 2,
        entanglement: str = "full",   # "full" ou "linear"
        phi_single: Optional[Callable[[float], float]] = None,
        phi_pair: Optional[Callable[[float, float], float]] = None,
    ):
        if reps < 1:
            raise ValueError("reps deve ser >= 1")
        self.reps = reps
        self.entanglement = entanglement
        self.phi_single = phi_single or (lambda t: t)
        self.phi_pair = phi_pair or (lambda a, b: (np.pi - a) * (np.pi - b))

    def _pairs(self, wires: Sequence[int]) -> Iterable[Tuple[int, int]]:
        return _pairs_linear(wires) if self.entanglement == "linear" else _pairs_full(wires)

    def build(self, x: np.ndarray, wires: Sequence[int]) -> None:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(wires):
            raise ValueError(f"Tamanho de x ({len(x)}) deve igualar nº de wires ({len(wires)}).")

        for w in wires:
            qml.Hadamard(wires=w)

        for _ in range(self.reps):
            for i, w in enumerate(wires):
                qml.RZ(2.0 * self.phi_single(x[i]), wires=w)

            for (wi, wj) in self._pairs(list(wires)):
                theta = 2.0 * self.phi_pair(x[wires.index(wi)], x[wires.index(wj)])
                qml.CNOT(wires=[wi, wj])
                qml.RZ(theta, wires=wj)
                qml.CNOT(wires=[wi, wj])
