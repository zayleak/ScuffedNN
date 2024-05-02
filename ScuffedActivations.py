from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ActivationFunction(ABC):

    @abstractmethod
    def activation(x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivActivation(x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):

    def activation(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def derivActivation(x: np.ndarray) -> np.ndarray:
        sig = Sigmoid.activation(x)
        return sig * (1 - sig)
    
class Identity(ActivationFunction):

    def activation(x: np.ndarray) -> np.ndarray:
        return x
    
    def derivActivation(x: np.ndarray) -> np.ndarray:
        return np.array(1)
