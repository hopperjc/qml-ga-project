"""Tests for the ansatz module — params_per_wire, init_weights, flatten roundtrip,
apply_ansatz layer count and per_layer_callback contract."""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
import pytest

from qml_ga.ansatz.base import (
    PARAMS_PER_WIRE,
    apply_ansatz,
    flatten_weights,
    init_weights,
    params_per_wire,
    unflatten_weights,
)


ANSATZE = list(PARAMS_PER_WIRE.keys())  # ['ansatz_1', ..., 'ansatz_6']


# ----------------------------- params_per_wire ------------------------------


def test_params_per_wire_matches_table():
    """params_per_wire(t) returns the value in the PARAMS_PER_WIRE dict."""
    for t, expected in PARAMS_PER_WIRE.items():
        assert params_per_wire(t) == expected


def test_params_per_wire_unknown_raises():
    with pytest.raises(ValueError, match="desconhecido"):
        params_per_wire("ansatz_42")


# ----------------------------- init_weights -------------------------------


@pytest.mark.parametrize("ansatz_type", ANSATZE)
def test_init_weights_shape(ansatz_type):
    """init_weights returns array of shape (depth, n_wires, P)."""
    depth, n_wires = 3, 4
    W = init_weights(ansatz_type, n_wires, depth, scale=0.1, seed=42)
    P = PARAMS_PER_WIRE[ansatz_type]
    assert tuple(W.shape) == (depth, n_wires, P)


@pytest.mark.parametrize("ansatz_type", ANSATZE)
def test_init_weights_deterministic(ansatz_type):
    """Same seed → same weights (across calls)."""
    W1 = init_weights(ansatz_type, n_wires=4, depth=3, scale=0.1, seed=42)
    W2 = init_weights(ansatz_type, n_wires=4, depth=3, scale=0.1, seed=42)
    assert np.allclose(np.array(W1), np.array(W2))


def test_init_weights_different_seeds_differ():
    """Different seeds → different weights."""
    W1 = init_weights("ansatz_1", n_wires=4, depth=3, scale=0.1, seed=1)
    W2 = init_weights("ansatz_1", n_wires=4, depth=3, scale=0.1, seed=2)
    assert not np.allclose(np.array(W1), np.array(W2))


@pytest.mark.parametrize("ansatz_type", ANSATZE)
def test_init_weights_respects_scale(ansatz_type):
    """All values in init_weights are within [-scale, scale]."""
    scale = 0.05
    W = init_weights(ansatz_type, n_wires=4, depth=3, scale=scale, seed=42)
    arr = np.array(W)
    assert arr.min() >= -scale - 1e-9
    assert arr.max() <= scale + 1e-9


# --------------------- flatten / unflatten roundtrip --------------------


@pytest.mark.parametrize("ansatz_type", ANSATZE)
def test_flatten_unflatten_roundtrip(ansatz_type):
    """unflatten(flatten(W)) == W in shape and values."""
    n_wires, depth = 4, 3
    W = init_weights(ansatz_type, n_wires, depth, seed=7)
    flat = flatten_weights(W)
    P = PARAMS_PER_WIRE[ansatz_type]
    assert flat.size == depth * n_wires * P
    W_back = unflatten_weights(flat, ansatz_type, n_wires, depth)
    assert np.allclose(np.array(W), np.array(W_back))


# ----------------------------- apply_ansatz -------------------------------


def _circuit_running(ansatz_type, depth, n_wires, callback=None):
    """Build a QNode that just runs apply_ansatz and measures Z_0."""
    dev = qml.device("default.qubit", wires=n_wires)
    wires = list(range(n_wires))

    @qml.qnode(dev)
    def circuit(W):
        apply_ansatz(ansatz_type, W, wires, depth, per_layer_callback=callback)
        return qml.expval(qml.PauliZ(0))

    return circuit


@pytest.mark.parametrize("ansatz_type", ANSATZE)
def test_apply_ansatz_runs_without_error(ansatz_type):
    """apply_ansatz produces a circuit that returns a finite real value."""
    depth, n_wires = 2, 4
    W = init_weights(ansatz_type, n_wires, depth, seed=42)
    circuit = _circuit_running(ansatz_type, depth, n_wires)
    out = float(circuit(W))
    assert np.isfinite(out)
    assert -1.0 <= out <= 1.0


@pytest.mark.parametrize("ansatz_type", ANSATZE)
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_apply_ansatz_callback_fires_depth_times(ansatz_type, depth):
    """per_layer_callback is invoked exactly `depth` times during a single run."""
    n_wires = 4
    W = init_weights(ansatz_type, n_wires, depth, seed=42)
    counter = {"calls": 0, "indices": []}

    def cb(d):
        counter["calls"] += 1
        counter["indices"].append(d)

    circuit = _circuit_running(ansatz_type, depth, n_wires, callback=cb)
    _ = float(circuit(W))
    assert counter["calls"] == depth
    assert counter["indices"] == list(range(depth))


def test_apply_ansatz_no_callback_default():
    """When per_layer_callback is omitted, no exception is raised (backwards-compat)."""
    depth, n_wires = 2, 4
    W = init_weights("ansatz_1", n_wires, depth, seed=42)
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(W):
        apply_ansatz("ansatz_1", W, list(range(n_wires)), depth)
        return qml.expval(qml.PauliZ(0))

    out = float(circuit(W))
    assert np.isfinite(out)


def test_apply_ansatz_wrong_shape_raises():
    """apply_ansatz must reject W with incompatible shape."""
    n_wires, depth = 4, 3
    W_wrong = qnp.zeros((depth, n_wires + 1, 3))  # wrong middle dim
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(W):
        apply_ansatz("ansatz_1", W, list(range(n_wires)), depth)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="Shape"):
        circuit(W_wrong)


def test_apply_ansatz_unknown_type_raises():
    """Unknown ansatz_type must raise inside apply_ansatz (not silently no-op)."""
    n_wires, depth = 4, 2
    W = qnp.zeros((depth, n_wires, 3))
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(W):
        apply_ansatz("ansatz_unknown", W, list(range(n_wires)), depth)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError):
        circuit(W)
