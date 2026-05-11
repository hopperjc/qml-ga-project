"""Tests for build_vqc — device selection, both feature maps, with/without noise."""
from __future__ import annotations

import numpy as np
import pytest

from qml_ga.ansatz.base import init_weights
from qml_ga.circuits.vqc import build_vqc


def _eval_once(ansatz_type, depth, n_qubits, fm, noise_config, x, seed=42):
    dev, circuit = build_vqc(ansatz_type, depth, n_qubits, feature_map=fm, noise_config=noise_config)
    W = init_weights(ansatz_type, n_qubits, depth, seed=seed)
    return float(circuit(W, x)), dev


def test_build_vqc_amplitude_no_noise_uses_default_qubit():
    x = np.array([1.0, 0.5, -0.5, 0.2])
    out, dev = _eval_once("ansatz_1", depth=2, n_qubits=2, fm="amplitude",
                          noise_config=None, x=x)
    assert dev.name == "default.qubit"
    assert -1.0 <= out <= 1.0


def test_build_vqc_zz_no_noise_uses_default_qubit():
    x = np.array([1.0, 0.5, -0.5, 0.2])
    out, dev = _eval_once("ansatz_1", depth=2, n_qubits=4, fm="zz",
                          noise_config=None, x=x)
    assert dev.name == "default.qubit"
    assert -1.0 <= out <= 1.0


def test_build_vqc_with_active_noise_switches_to_default_mixed():
    x = np.array([1.0, 0.5, -0.5, 0.2])
    nc = {"after_feature_map": {"type": "amplitude_damping", "gamma": 0.05}}
    out, dev = _eval_once("ansatz_1", depth=2, n_qubits=2, fm="amplitude",
                          noise_config=nc, x=x)
    assert dev.name == "default.mixed"
    assert -1.0 <= out <= 1.0


def test_build_vqc_with_zero_gamma_keeps_default_qubit():
    """Noise block with gamma=0 in every slot is equivalent to no noise — must
    NOT switch to default.mixed (would needlessly slow down the experiment)."""
    x = np.array([1.0, 0.5, -0.5, 0.2])
    nc = {"after_feature_map": {"type": "amplitude_damping", "gamma": 0.0}}
    out, dev = _eval_once("ansatz_1", depth=2, n_qubits=2, fm="amplitude",
                          noise_config=nc, x=x)
    assert dev.name == "default.qubit"
    assert -1.0 <= out <= 1.0


def test_build_vqc_unknown_fm_raises():
    x = np.array([1.0, 0.5, -0.5, 0.2])
    with pytest.raises(ValueError, match="Feature map desconhecido"):
        _eval_once("ansatz_1", depth=2, n_qubits=2, fm="bogus", noise_config=None, x=x)
