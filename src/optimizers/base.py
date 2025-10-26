from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class BaseOptimizer(ABC):
    @abstractmethod
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        ...
