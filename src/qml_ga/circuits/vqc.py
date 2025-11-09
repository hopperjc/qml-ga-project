import pennylane as qml
from qml_ga.ansatz.base import apply_ansatz
from qml_ga.feature_maps.amplitude_encoding import amplitude_embed
from qml_ga.feature_maps.zz_feature_map import zz_feature_map

def build_vqc(ansatz_type: str, depth: int, n_qubits: int, feature_map: str = "amplitude", shots=None):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    wires = list(range(n_qubits))
    fm = feature_map.lower()

    @qml.qnode(dev)
    def circuit(W, x):
        # x é 1D
        if fm == "amplitude":
            amplitude_embed(x, wires=wires)
        elif fm == "zz":
            zz_feature_map(x, wires=wires, with_h=True)
        else:
            raise ValueError(f"Feature map desconhecido: {feature_map}")

        apply_ansatz(ansatz_type, W, wires, depth)
        return qml.expval(qml.PauliZ(wires[0]))

    return dev, circuit
