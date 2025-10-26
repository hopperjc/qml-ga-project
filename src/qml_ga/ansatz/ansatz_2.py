# src/qml_ga/ansatz/ansatz2.py
import pennylane as qml

def ansatz_2(W_slice, wires):
    """
    P = 3 por wire: [RX, RZ, CRZ-coupling]
    - rotações locais RX,RZ
    - acoplamento CRZ alternado ímpar e par em linhas separadas (padrão em escada)
    """
    n = len(wires)
    # rotações por wire
    for i, w in enumerate(wires):
        qml.RX(W_slice[i, 0], wires=w)
        qml.RZ(W_slice[i, 1], wires=w)

    # CRZ de pares ímpares
    for j in range(n - 1, 0, -1):
        if j % 2 != 0:
            qml.CRZ(W_slice[j, 2], wires=(wires[j], wires[j - 1]))
    # CRZ de pares pares
    for k in range(n - 1, 0, -1):
        if k % 2 == 0:
            qml.CRZ(W_slice[k, 2], wires=(wires[k], wires[k - 1]))
