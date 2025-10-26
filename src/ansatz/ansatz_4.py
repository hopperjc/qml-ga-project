from typing import Sequence, Tuple
import pennylane as qml
from .base import IAnsatz

class Layer4Ansatz(IAnsatz):
    """
    layer_4 do notebook:
      - Para cada qubit i: RX(W[d,i,0]); RZ(W[d,i,1])
      - Todos-para-todos com CRX: para j!=k, CRX(W[d,j,2]) em (j -> k)
      - Depois: para cada l: RX(W[d,l,3]); RZ(W[d,l,4])
    """
    def __init__(self, depth: int = 1):
        self.depth = int(depth)

    def param_shape(self, n_wires: int) -> Tuple[int, ...]:
        return (self.depth, n_wires, 5)

    def num_params(self, n_wires: int) -> int:
        d, n, k = self.param_shape(n_wires)
        return d * n * k

    def apply(self, weights, wires: Sequence[int]) -> None:
        n = len(wires)
        D, N, _ = self.param_shape(n)
        W = self._unflatten(weights, (D, N, 5))

        for d in range(D):
            for i, w in enumerate(wires):
                qml.RX(W[d, i, 0], wires=w)
                qml.RZ(W[d, i, 1], wires=w)
            for j in range(n - 1, -1, -1):
                for k in range(n - 1, -1, -1):
                    if j != k:
                        qml.CRX(W[d, j, 2], wires=(wires[j], wires[k]))
            for l, w in enumerate(wires):
                qml.RX(W[d, l, 3], wires=w)
                qml.RZ(W[d, l, 4], wires=w)
