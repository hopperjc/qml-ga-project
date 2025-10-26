from qml_ga.ansatz.base import IAnsatz
from qml_ga.ansatz.ansatz_1 import Layer1Ansatz
from qml_ga.ansatz.ansatz_2 import Layer2Ansatz
from qml_ga.ansatz.ansatz_3 import Layer3Ansatz
from qml_ga.ansatz.ansatz_4 import Layer4Ansatz
from qml_ga.ansatz.ansatz_5 import Layer5Ansatz
from qml_ga.ansatz.ansatz_6 import Layer6Ansatz

REGISTRY = {
    "ansatz_1": Layer1Ansatz,
    "ansatz_2": Layer2Ansatz,
    "ansatz_3": Layer3Ansatz,
    "ansatz_4": Layer4Ansatz,
    "ansatz_5": Layer5Ansatz,
    "ansatz_6": Layer6Ansatz,
}

__all__ = [
    "IAnsatz",
    "Layer1Ansatz",
    "Layer2Ansatz",
    "Layer3Ansatz",
    "Layer4Ansatz",
    "Layer5Ansatz",
    "Layer6Ansatz",
    "REGISTRY",
]
