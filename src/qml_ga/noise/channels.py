"""Single-qubit noise channel primitives for PennyLane density-matrix devices.

These wrap qml.AmplitudeDamping / qml.PhaseDamping / qml.DepolarizingChannel
to apply on a list of wires uniformly. They MUST be invoked inside a QNode
that uses a mixed-state device (e.g. default.mixed); calling them on
default.qubit raises a runtime error from PennyLane.

Reference (Kraus operators):
    Amplitude damping (γ):
        K0 = [[1, 0], [0, sqrt(1-γ)]]
        K1 = [[0, sqrt(γ)], [0, 0]]
    Phase damping (γ):
        K0 = [[1, 0], [0, sqrt(1-γ)]]
        K1 = [[0, 0], [0, sqrt(γ)]]
    Depolarizing (p):
        K0 = sqrt(1-p) I
        K1 = sqrt(p/3) X, K2 = sqrt(p/3) Y, K3 = sqrt(p/3) Z
"""
from __future__ import annotations

from typing import Iterable

import pennylane as qml


def _validate_gamma(gamma: float) -> float:
    g = float(gamma)
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"gamma must be in [0, 1]; got {g}")
    return g


def apply_amplitude_damping(wires: Iterable[int], gamma: float) -> None:
    """Apply amplitude damping with rate γ on every wire in `wires`."""
    g = _validate_gamma(gamma)
    if g == 0.0:
        return
    for w in wires:
        qml.AmplitudeDamping(g, wires=w)


def apply_phase_damping(wires: Iterable[int], gamma: float) -> None:
    """Apply phase damping with rate γ on every wire in `wires`."""
    g = _validate_gamma(gamma)
    if g == 0.0:
        return
    for w in wires:
        qml.PhaseDamping(g, wires=w)


def apply_depolarizing(wires: Iterable[int], p: float) -> None:
    """Apply depolarizing channel with probability p on every wire in `wires`."""
    g = _validate_gamma(p)
    if g == 0.0:
        return
    for w in wires:
        qml.DepolarizingChannel(g, wires=w)
