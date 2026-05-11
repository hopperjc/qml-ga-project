"""Noise modeling for QML-GA experiments.

Public API:
    apply_noise_layer(wires, cfg) — dispatcher used by build_vqc

Channel primitives in `channels.py` wrap PennyLane non-unitary operations and
are only valid on density-matrix devices (default.mixed). When the noise config
is None, none of these are reachable, and build_vqc falls back to default.qubit.
"""
from qml_ga.noise.channels import (
    apply_amplitude_damping,
    apply_phase_damping,
    apply_depolarizing,
)
from qml_ga.noise.policy import apply_noise_layer, requires_mixed_device

__all__ = [
    "apply_amplitude_damping",
    "apply_phase_damping",
    "apply_depolarizing",
    "apply_noise_layer",
    "requires_mixed_device",
]
