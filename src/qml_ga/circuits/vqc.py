from typing import Any, Dict, Optional

import pennylane as qml

from qml_ga.ansatz.base import apply_ansatz
from qml_ga.feature_maps.amplitude_encoding import amplitude_embed
from qml_ga.feature_maps.zz_feature_map import zz_feature_map
from qml_ga.noise.policy import apply_noise_layer, requires_mixed_device


def build_vqc(
    ansatz_type: str,
    depth: int,
    n_qubits: int,
    feature_map: str = "amplitude",
    shots: Optional[int] = None,
    noise_config: Optional[Dict[str, Any]] = None,
):
    """Build the VQC device + QNode pair.

    `noise_config` (optional) has shape:
        {"after_feature_map": {"type": <kind>, "gamma": <float>},
         "per_ansatz_layer": {"type": <kind>, "gamma": <float>}}
    Either or both keys may be absent; missing/zero entries are no-ops.

    When any noise slot is active, the device is `default.mixed` (density
    matrix; ~2× slower, supports non-unitary channels). Otherwise the device
    stays `default.qubit` for backwards compatibility with existing experiments.
    """
    use_mixed = requires_mixed_device(noise_config)
    dev_name = "default.mixed" if use_mixed else "default.qubit"
    dev = qml.device(dev_name, wires=n_qubits, shots=shots)
    wires = list(range(n_qubits))
    fm = feature_map.lower()

    nc = noise_config or {}
    after_fm_cfg = nc.get("after_feature_map")
    per_layer_cfg = nc.get("per_ansatz_layer")

    def _per_layer_noise(_d):
        apply_noise_layer(wires, per_layer_cfg)

    per_layer_cb = _per_layer_noise if per_layer_cfg else None

    @qml.qnode(dev)
    def circuit(W, x):
        if fm == "amplitude":
            amplitude_embed(x, wires=wires)
        elif fm == "zz":
            zz_feature_map(x, wires=wires, with_h=True)
        else:
            raise ValueError(f"Feature map desconhecido: {feature_map}")

        apply_noise_layer(wires, after_fm_cfg)
        apply_ansatz(ansatz_type, W, wires, depth, per_layer_callback=per_layer_cb)
        return qml.expval(qml.PauliZ(wires[0]))

    return dev, circuit
