# src/qml_ga/ansatz/ansatz4.py
import pennylane as qml

def ansatz_4(W_slice, wires):
    """
    P = 5 por wire: [RX, RZ, CRX-coupling, RX2, RZ2]
    - camada RX,RZ local
    - todos-contra-todos com CRX (parâmetro indexado pelo 'j' de controle)
    - outra camada RX,RZ local
    """
    n = len(wires)
    # primeira passada RX,RZ
    for i, w in enumerate(wires):
        qml.RX(W_slice[i, 0], wires=w)
        qml.RZ(W_slice[i, 1], wires=w)
    # CRX all-to-all (j controla, k alvo; j!=k)
    for j in range(n - 1, -1, -1):
        for k in range(n - 1, -1, -1):
            if j != k:
                qml.CRX(W_slice[j, 2], wires=(wires[j], wires[k]))
    # segunda passada RX,RZ
    for i, w in enumerate(wires):
        qml.RX(W_slice[i, 3], wires=w)
        qml.RZ(W_slice[i, 4], wires=w)
