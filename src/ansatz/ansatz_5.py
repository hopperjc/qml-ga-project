from typing import Sequence, Tuple
import pennylane as qml
from .base import IAnsatz

class Layer5Ansatz(IAnsatz):
    """
    layer_5 do notebook:
      - Para cada qubit i: RY(W[d,i,0]); RZ(W[d,i,1])
      - CNOTs em pares descendentes somente quando j ímpar: (j -> j-1), j = n-1..1
      - Para k = 0 .. n-3: aplica RY(W[d,k,3]) e RZ(W[d,k,4]) em (k+1)  [note: não aplica nos extremos]
      - Para l = 0 .. n-3: se l+2 != n-1, CNOT((l+2) -> (l+1))  [pula o par cujo controle seria o último qubit]
    Observação: esta camada usa colunas 0,1,3,4; a coluna 2 é ignorada por compatibilidade de shape (5).
    """
    def __init__(self, depth: int = 1):
        self.depth = int(depth)

    def param_shape(self, n_wires: int) -> Tuple[int, ...]:
        # shape com 5 colunas porque a camada usa índices [0,1,3,4]
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
                qml.RY(W[d, i, 0], wires=w)
                qml.RZ(W[d, i, 1], wires=w)

            if n >= 2:
                for j in range(n - 1, 0, -1):
                    if j % 2 != 0:
                        qml.CNOT(wires=(wires[j], wires[j - 1]))

            if n >= 3:
                for k in range(0, n - 2):
                    qml.RY(W[d, k, 3], wires=wires[k + 1])
                    qml.RZ(W[d, k, 4], wires=wires[k + 1])

                for l in range(0, n - 2):
                    if (l + 2) != (n - 1):
                        qml.CNOT(wires=(wires[l + 2], wires[l + 1]))
