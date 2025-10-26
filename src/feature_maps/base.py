from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class IFeatureMap(ABC):
    @abstractmethod
    def build(self, x: np.ndarray, wires: Sequence[int]) -> None:
        ...
