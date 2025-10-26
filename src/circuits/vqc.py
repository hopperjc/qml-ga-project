import numpy as np
import pennylane as qml

class VQC:
    def __init__(self, device, feature_map, ansatz):
        self.dev = device
        self.fm = feature_map
        self.ans = ansatz
        self.wires = list(range(device.num_wires))

    def qnode(self):
        @qml.qnode(self.dev)
        def _circ(weights, x):
            self.fm.build(x, wires=self.wires)
            self.ans.apply(weights, wires=self.wires)
            return qml.expval(qml.PauliZ(self.wires[0]))
        return _circ

    def predict_proba(self, weights, X):
        circ = self.qnode()
        outs = np.array([circ(weights, x) for x in X])  # [-1, 1]
        p1 = 0.5 * (1 - outs)
        return np.vstack([1 - p1, p1]).T

    def predict(self, weights, X, threshold=0.5):
        return (self.predict_proba(weights, X)[:, 1] >= threshold).astype(int)
