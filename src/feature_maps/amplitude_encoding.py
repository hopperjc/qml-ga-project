import numpy as np
import pennylane as qml
from typing import Sequence
from .base import IFeatureMap

def _pad_to_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=float).ravel()
    if len(v) == dim:
        return v
    out = np.zeros(dim, dtype=float)
    out[: min(len(v), dim)] = v[: min(len(v), dim)]
    return out

class AmplitudeEncoding(IFeatureMap):
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def build(self, x: np.ndarray, wires: Sequence[int]) -> None:
        dim = 2 ** len(wires)
        xv = _pad_to_dim(x, dim)
        qml.AmplitudeEmbedding(xv, wires=wires, normalize=self.normalize)
