from abc import ABC, abstractmethod
import numpy as np
import ScuffedWeightInit

class LinearLayer(ABC):
    def __init__(self, inputLayerSize: int, outputLayerSize: int, hasBias: bool = True, weightInit: ScuffedWeightInit.ScuffedWeights = ScuffedWeightInit.Standard) -> None:
        self.weightInit = weightInit
        self.__hasBias = hasBias
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.initWeights()
        
    def initWeights(self):
        self.__weights = self.weightInit.getWeights(self.inputLayerSize, self.outputLayerSize)
        if self.__hasBias:
            self.__weights = np.vstack((np.zeros((1, self.outputLayerSize)), self.__weights))
    
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
