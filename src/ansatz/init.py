from .base import IAnsatz
from .ansatz_1 import Layer1Ansatz
from .ansatz_2 import Layer2Ansatz
from .ansatz_3 import Layer3Ansatz
from .ansatz_4 import Layer4Ansatz
from .ansatz_5 import Layer5Ansatz
from .ansatz_6 import Layer6Ansatz

REGISTRY = {
    "layer_1": Layer1Ansatz,
    "layer_2": Layer2Ansatz,
    "layer_3": Layer3Ansatz,
    "layer_4": Layer4Ansatz,
    "layer_5": Layer5Ansatz,
    "layer_6": Layer6Ansatz,
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
