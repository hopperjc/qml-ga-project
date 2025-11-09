import pennylane as qml
import numpy as np

def zz_feature_map(x, wires, alpha=2.0, with_h=False):
    """
    'ZZFeatureMap' estilo Qiskit: RZ(x_i) + e^{i*alpha*x_i*x_j Z_i Z_j}.
    Opcional: camadas de Hadamard antes/depois (with_h=True).
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) != len(wires):
        raise ValueError(f"ZZFeatureMap requer len(x)==len(wires); recebi {len(x)} e {len(wires)}")
    n = len(wires)
    if with_h:
        for w in wires:
            qml.Hadamard(wires=w)
    # unários
    for i, w in enumerate(wires):
        qml.RZ(x[i], wires=w)
    # binários ZZ
    for i in range(n):
        for j in range(i + 1, n):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RZ(alpha * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])
    # if with_h:
    #     for w in wires:
    #         qml.Hadamard(wires=w)
