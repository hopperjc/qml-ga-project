from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import numpy as np

class IAnsatz(ABC):
    @abstractmethod
    def num_params(self, n_wires: int) -> int:
        ...

    @abstractmethod
    def param_shape(self, n_wires: int) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def apply(self, weights, wires: Sequence[int]) -> None:
        ...

    # utilitário comum
    def _unflatten(self, weights, shape: Tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(weights, dtype=float).ravel()
        need = int(np.prod(shape))
        if arr.size != need:
            raise ValueError(f"Tamanho de pesos incompatível: recebido {arr.size}, esperado {need} para shape {shape}.")
        return arr.reshape(shape)
