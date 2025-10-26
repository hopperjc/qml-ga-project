import json
import os
import pennylane as qml

def specs_snapshot(qnode, weights, x_sample, out_path: str):
    sp = qml.specs(qnode)(weights, x_sample)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2, default=str)
