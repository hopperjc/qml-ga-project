"""Noise application policy: dispatches a config dict to the right channel.

Config schema (used inside feature_map YAMLs):

    noise:
      after_feature_map:
        type: amplitude_damping | phase_damping | depolarizing | none
        gamma: <float in [0, 1]>
      per_ansatz_layer:
        type: ...
        gamma: ...

Both keys are optional. If both are absent (or both are `type: none`), the
circuit runs noise-free and the device falls back to default.qubit.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from qml_ga.noise.channels import (
    apply_amplitude_damping,
    apply_depolarizing,
    apply_phase_damping,
)

_DISPATCH = {
    "amplitude_damping": apply_amplitude_damping,
    "phase_damping": apply_phase_damping,
    "depolarizing": apply_depolarizing,
}


def apply_noise_layer(wires: Iterable[int], cfg: Optional[Dict[str, Any]]) -> None:
    """Apply one noise layer described by `cfg`. Silently no-op if cfg is None,
    `type` is missing/'none', or gamma is 0."""
    if not cfg:
        return
    ntype = str(cfg.get("type", "none")).lower()
    if ntype in ("none", ""):
        return
    if ntype not in _DISPATCH:
        raise ValueError(f"Unknown noise type: {ntype!r}. Valid: {list(_DISPATCH)}")
    gamma = float(cfg.get("gamma", 0.0))
    _DISPATCH[ntype](wires, gamma)


def _slot_is_active(slot: Optional[Dict[str, Any]]) -> bool:
    if not slot:
        return False
    ntype = str(slot.get("type", "none")).lower()
    if ntype in ("none", ""):
        return False
    return float(slot.get("gamma", 0.0)) > 0.0


def requires_mixed_device(noise_cfg: Optional[Dict[str, Any]]) -> bool:
    """Return True iff the noise config has at least one slot with a non-zero
    rate. The pure-state device (default.qubit) cannot simulate non-unitary
    channels, so the caller must use default.mixed in that case."""
    if not noise_cfg:
        return False
    for slot_key in ("after_feature_map", "per_ansatz_layer"):
        if _slot_is_active(noise_cfg.get(slot_key)):
            return True
    return False
