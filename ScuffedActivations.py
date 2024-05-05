from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ActivationFunction(ABC):

    @abstractmethod
    def activation(self, x: np.ndarray) -> np.ndarray:
        pass

    # this should be the deriviate with the layers outputs as input
    # ex d/dx sigmoid(x) = sigmoid(x)(1-sigmoid(x)) OR
    # this should be the deriviate with the layers transform as input (i.e before the activation of a layer)
    @abstractmethod
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):

    def activation(self, x: np.ndarray) -> np.ndarray:
        # clipping if it blows up (computing np.exp(-x), xi might have very high entries)
        x = np.clip(x, -100, 100) 
        return 1 / (1 + np.exp(-x))
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return layerOutputs * (1 - layerOutputs)

# should really only use for the final layer otherwise weights will blow up
class Identity(ActivationFunction):

    def activation(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return np.ones(layerOutputs.shape)

class SoftMax(ActivationFunction):

    def activation(self, x: np.ndarray) -> np.ndarray:
        max = np.max(x)
        eVec = np.exp(x - max)
        return eVec / np.sum(eVec)
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        pass

class Tanh(ActivationFunction):

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return 1 - np.square(layerOutputs)

class LeakyReLU(ActivationFunction):

    def __init__(self, leakParam: float = 0.01) -> None:
        self.leakParam = leakParam

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.leakParam*x)

    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return np.where(layerTransforms > 0, 1, self.leakParam)
    
class ReLU(ActivationFunction):

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return np.where(layerTransforms > 0, 1, 0)
    
class ELU(ActivationFunction):

    def __init__(self, eluParam: float = 0.01) -> None:
        self.eluParam = eluParam

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.eluParam * (np.exp(x) - 1))
    
    def derivActivation(self, layerOutputs: np.ndarray, layerTransforms: np.ndarray) -> np.ndarray:
        return np.where(layerTransforms > 0, 1, self.activation(layerTransforms) + self.eluParam)

