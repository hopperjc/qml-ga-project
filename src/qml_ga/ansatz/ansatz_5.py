import pennylane as qml

def ansatz_5(W_slice, wires):
    """
    P = 5 por wire: [RY, RZ, (pad), RY2, RZ2]
    - RY,RZ locais
    - CNOTs em pares ímpares (padrão escada)
    - Nova passada RY,RZ em qubits 1..n-2 (índices deslocados como no seu código)
    - Mais CNOTs (padrão deslocado), preservando a lógica original
    """
    n = len(wires)

    # primeira passada RY,RZ local
    for i, w in enumerate(wires):
        qml.RY(W_slice[i, 0], wires=w)
        qml.RZ(W_slice[i, 1], wires=w)

    # CNOT em pares ímpares (j ímpar controla j -> j-1)
    for j in range(n - 1, 0, -1):
        if j % 2 != 0:
            qml.CNOT(wires=(wires[j], wires[j - 1]))

    # segunda passada RY,RZ em qubits 1..n-2 (k+1)
    # (mantém a escolha original k in [0..n-3] aplicando no wire k+1)
    for k in range(0, max(0, n - 2)):
        qml.RY(W_slice[k, 3], wires=wires[k + 1])
        qml.RZ(W_slice[k, 4], wires=wires[k + 1])

    # CNOT em padrão deslocado (l+2 -> l+1), exceto quando l+2 == n-1
    for l in range(0, max(0, n - 2)):
        if (l + 2) != (n - 1):
            qml.CNOT(wires=(wires[l + 2], wires[l + 1]))
