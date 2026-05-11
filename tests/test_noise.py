"""Tests for the noise module — correctness is paramount here.

Critical invariants validated:
  1. γ=0 on default.mixed must give identical expectation (up to numerical noise)
     to default.qubit with no noise.
  2. γ=1 amplitude damping must collapse every qubit to |0⟩, so ⟨Z₀⟩ = +1.
  3. γ=1 phase damping leaves Z eigenvalues unchanged: ⟨Z₀⟩ on |+⟩ → 0,
     on |0⟩ → +1.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qml_ga.noise.channels import (
    apply_amplitude_damping,
    apply_depolarizing,
    apply_phase_damping,
)
from qml_ga.noise.policy import apply_noise_layer, requires_mixed_device


# ----------------------------- policy ------------------------------


def test_requires_mixed_device_returns_false_for_none():
    assert requires_mixed_device(None) is False


def test_requires_mixed_device_returns_false_for_empty():
    assert requires_mixed_device({}) is False


def test_requires_mixed_device_returns_false_for_zero_gamma():
    cfg = {"after_feature_map": {"type": "amplitude_damping", "gamma": 0.0}}
    assert requires_mixed_device(cfg) is False


def test_requires_mixed_device_returns_false_for_type_none():
    cfg = {"after_feature_map": {"type": "none", "gamma": 0.5}}
    assert requires_mixed_device(cfg) is False


def test_requires_mixed_device_returns_true_with_active_slot():
    cfg = {"per_ansatz_layer": {"type": "phase_damping", "gamma": 0.01}}
    assert requires_mixed_device(cfg) is True


def test_apply_noise_layer_unknown_type_raises():
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circ():
        apply_noise_layer([0], {"type": "bogus", "gamma": 0.5})
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="Unknown noise type"):
        circ()


def test_apply_noise_layer_invalid_gamma_raises():
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circ():
        apply_noise_layer([0], {"type": "amplitude_damping", "gamma": 1.5})
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="gamma must be in"):
        circ()


# ------------------------ idempotency at γ=0 ------------------------


def _expval_z0_with_h(device_name: str, channel_fn=None, gamma=0.0):
    """Prepare |+⟩ on a 1-qubit device, optionally apply a channel, return ⟨Z₀⟩."""
    dev = qml.device(device_name, wires=1)

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(wires=0)
        if channel_fn is not None:
            channel_fn([0], gamma)
        return qml.expval(qml.PauliZ(0))

    return float(circ())


def test_amplitude_damping_gamma_zero_matches_pure():
    pure = _expval_z0_with_h("default.qubit")
    mixed_zero = _expval_z0_with_h("default.mixed", apply_amplitude_damping, gamma=0.0)
    assert abs(pure - mixed_zero) < 1e-9


def test_phase_damping_gamma_zero_matches_pure():
    pure = _expval_z0_with_h("default.qubit")
    mixed_zero = _expval_z0_with_h("default.mixed", apply_phase_damping, gamma=0.0)
    assert abs(pure - mixed_zero) < 1e-9


def test_depolarizing_p_zero_matches_pure():
    pure = _expval_z0_with_h("default.qubit")
    mixed_zero = _expval_z0_with_h("default.mixed", apply_depolarizing, gamma=0.0)
    assert abs(pure - mixed_zero) < 1e-9


# ------------------------ extreme channels ------------------------


def test_amplitude_damping_gamma_one_collapses_to_ground():
    """Amplitude damping with γ=1 sends any state to |0⟩, so ⟨Z₀⟩ = +1."""
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(wires=0)  # |+⟩
        apply_amplitude_damping([0], gamma=1.0)
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circ()) - 1.0) < 1e-9


def test_phase_damping_gamma_one_dephases_plus_to_zero_z():
    """Phase damping with γ=1 fully dephases; |+⟩⟨+| → 0.5 I, so ⟨Z₀⟩ = 0."""
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(wires=0)
        apply_phase_damping([0], gamma=1.0)
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circ())) < 1e-9


def test_phase_damping_preserves_z_eigenstates():
    """|0⟩ is a Z-eigenstate; phase damping commutes with Z, so ⟨Z₀⟩ stays at +1."""
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circ():
        # state stays |0⟩
        apply_phase_damping([0], gamma=0.5)
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circ()) - 1.0) < 1e-9


# ------------------------ end-to-end via build_vqc ------------------------


def test_build_vqc_gamma_zero_matches_clean():
    """A circuit built with gamma=0 noise must give the same expectation as the
    clean (default.qubit) circuit, regardless of the noise type configured."""
    from qml_ga.ansatz.base import init_weights
    from qml_ga.circuits.vqc import build_vqc

    x = np.array([1.0, 0.5, -0.5, 0.2])
    n_qubits = 2
    W = init_weights("ansatz_1", n_qubits, depth=2, seed=42)

    _, clean = build_vqc("ansatz_1", 2, n_qubits, feature_map="amplitude",
                         noise_config=None)
    _, noisy_zero = build_vqc(
        "ansatz_1", 2, n_qubits, feature_map="amplitude",
        noise_config={
            "after_feature_map": {"type": "amplitude_damping", "gamma": 0.0},
            "per_ansatz_layer":  {"type": "amplitude_damping", "gamma": 0.0},
        },
    )

    # Both should pass through default.qubit (zero gamma → no mixed)
    out_clean = float(clean(W, x))
    out_zero = float(noisy_zero(W, x))
    assert abs(out_clean - out_zero) < 1e-9
