from abc import ABC, abstractmethod
import numpy as np

class ScuffedWeights(ABC):

    @abstractmethod
    def getWeights(inputLayerSize: int, outputLayerSize: int) -> np.ndarray:
        pass

class Standard(ScuffedWeights):

    def getWeights(inputLayerSize: int, outputLayerSize: int) -> np.ndarray:
        return 2*np.random.randn(inputLayerSize, outputLayerSize) - 1
    
class Xavier(ScuffedWeights):

    def getWeights(inputLayerSize: int, outputLayerSize: int) -> np.ndarray:
        return np.random.randn(inputLayerSize, outputLayerSize) / np.sqrt(inputLayerSize)

class He(ScuffedWeights):

    def getWeights(inputLayerSize: int, outputLayerSize: int) -> np.ndarray:
        return np.random.randn(inputLayerSize, outputLayerSize) * np.sqrt(2/inputLayerSize)

    
    
