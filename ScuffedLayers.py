from abc import ABC, abstractmethod
import numpy as np

class LinearLayer(ABC):
    def __init__(self, inputLayerSize: int, outputLayerSize: int, hasBias: bool = True) -> None:
        self.__weights = np.random.randn(inputLayerSize, outputLayerSize) / (np.sqrt(inputLayerSize))
        self.__hasBias = hasBias
        if hasBias:
            self.__weights = np.vstack((np.zeros((1, outputLayerSize)), self.__weights))
    
    @property
    def hasBias(self) -> bool:
        return self.__hasBias
    
    @property
    def hasBias(self) -> bool:
        return self.__hasBias

    @property
    def weights(self) -> np.ndarray:
        return self.__weights
    
    def forward(self, x : np.ndarray) -> np.ndarray:
        return np.dot(x, self.__weights)
    
    def gradientUpdate(self, alpha: int, avgPartialDerivs: np.ndarray):
        self.__weights += alpha * avgPartialDerivs
