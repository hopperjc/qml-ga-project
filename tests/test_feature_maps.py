"""Smoke tests for the two feature maps."""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qml_ga.feature_maps.amplitude_encoding import amplitude_embed
from qml_ga.feature_maps.zz_feature_map import zz_feature_map


def test_amplitude_embed_runs_on_4_features():
    n_qubits = 2  # ceil(log2(4))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        amplitude_embed(x, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    x = np.array([1.0, 0.5, -0.5, 0.2])
    out = float(circuit(x))
    assert -1.0 <= out <= 1.0


def test_zz_feature_map_runs_on_4_features():
    n_qubits = 4
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        zz_feature_map(x, wires=range(n_qubits), with_h=True)
        return qml.expval(qml.PauliZ(0))

    x = np.array([1.0, 0.5, -0.5, 0.2])
    out = float(circuit(x))
    assert -1.0 <= out <= 1.0
