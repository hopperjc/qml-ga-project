from typing import Sequence, Tuple
import pennylane as qml
from .base import IAnsatz

class Layer6Ansatz(IAnsatz):
    """
    layer_6 do notebook (com correção de bug no índice do parâmetro):
      - Para cada qubit i: RY(W[d,i,0]); RZ(W[d,i,1])
      - Para j = n-1..1: CRZ(W[d,j,2]) em (j -> j-1)
    """
    def __init__(self, depth: int = 1):
        self.depth = int(depth)

    def param_shape(self, n_wires: int) -> Tuple[int, ...]:
        return (self.depth, n_wires, 3)

    def num_params(self, n_wires: int) -> int:
        d, n, k = self.param_shape(n_wires)
        return d * n * k

    def apply(self, weights, wires: Sequence[int]) -> None:
        n = len(wires)
        D, N, _ = self.param_shape(n)
        W = self._unflatten(weights, (D, N, 3))

        for d in range(D):
            for i, w in enumerate(wires):
                qml.RY(W[d, i, 0], wires=w)
                qml.RZ(W[d, i, 1], wires=w)
            if n >= 2:
                for j in range(n - 1, 0, -1):
                    qml.CRZ(W[d, j, 2], wires=(wires[j], wires[j - 1]))
