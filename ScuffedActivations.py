from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class ActivationFunction(ABC):

    @abstractmethod
    def activation(x: np.ndarray) -> np.ndarray:
        pass

    # this should be the deriviate with the layers outputs as input
    # ex d/dx sigmoid(x) = sigmoid(x)(1-sigmoid(x))
    @abstractmethod
    def derivActivation(layerOutputs: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):

    def activation(x: np.ndarray) -> np.ndarray:
        # clipping if it blows up (computing np.exp(-x), x might have very high entries)
        x = np.clip(x, -100, 100) 
        return 1 / (1 + np.exp(-x))
    
    def derivActivation(layerOutputs: np.ndarray) -> np.ndarray:
        return layerOutputs * (1 - layerOutputs)

# should really only use for the final layer otherwise weights will blow up
class Identity(ActivationFunction):

    def activation(x: np.ndarray) -> np.ndarray:
        return x
    
    def derivActivation(layerOutputs: np.ndarray) -> np.ndarray:
        return np.ones(layerOutputs.shape)

class SoftMax(ActivationFunction):

    def activation(x: np.ndarray) -> np.ndarray:
        max = np.max(x)
        eVec = np.exp(x - max)
        return eVec / np.sum(eVec)
    
    def derivActivation(layerOutputs: np.ndarray) -> np.ndarray:
        pass
        # id = np.identity(max(layerOutputs.shape[0], layerOutputs.shape[1]))

        # return softMax * (1 - softMax)