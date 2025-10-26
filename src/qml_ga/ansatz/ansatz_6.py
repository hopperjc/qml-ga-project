# src/qml_ga/ansatz/ansatz6.py
import pennylane as qml

def ansatz_6(W_slice, wires):
    """
    P = 3 por wire: [RY, RZ, CRZ-coupling]
    - RY,RZ locais
    - acoplamento CRZ em cadeia descendente (j -> j-1)
    (corrige bug: usar W_slice[j, 2] no laço, não W_slice[i, 2])
    """
    n = len(wires)
    # rotações locais
    for i, w in enumerate(wires):
        qml.RY(W_slice[i, 0], wires=w)
        qml.RZ(W_slice[i, 1], wires=w)
    # cadeia CRZ descendente
    for j in range(n - 1, 0, -1):
        qml.CRZ(W_slice[j, 2], wires=(wires[j], wires[j - 1]))
