import pennylane as qml

def ansatz_1(W_slice, wires):
    n = len(wires)
    # rotações por wire
    for i, w in enumerate(wires):
        qml.RX(W_slice[i, 0], wires=w)
        qml.RZ(W_slice[i, 1], wires=w)
    # acoplamentos CRX em anel
    qml.CRX(W_slice[0, 2], wires=(wires[-1], wires[0]))
    for j in range(n - 1, 0, -1):
        qml.CRX(W_slice[j, 2], wires=(wires[j], wires[j - 1]))