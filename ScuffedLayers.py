from abc import ABC, abstractmethod
from typing import List
import numpy as np
import ScuffedWeightInit
import ScuffedRegularizations

class LinearLayer():
    def __init__(self, inputLayerSize: int, outputLayerSize: int, hasBias: bool = True, weightInit: ScuffedWeightInit.ScuffedWeights = ScuffedWeightInit.Standard) -> None:
        self.weightInit = weightInit
        self.__hasBias = hasBias
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.initWeights()
        
    def initWeights(self) -> None:
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
    
    def backpropStep(self, 
                     alpha: int, 
                     avgPartialDerivs: np.ndarray, 
                     regularizations: List[ScuffedRegularizations.RegularizationMethod], 
                     numTraining: int) -> None:
        self.__weights += alpha * avgPartialDerivs
        for regularization in regularizations:
            self.__weights += regularization.computeGradientCost(self.__weights, numTraining)

class DropoutLayer(LinearLayer):

    #neuron i has a dropoutRate chance of being eliminated at epoch
    def __init__(self, 
                 inputLayerSize: int, 
                 outputLayerSize: int, 
                 hasBias: bool = True, 
                 weightInit: ScuffedWeightInit.ScuffedWeights = ScuffedWeightInit.Standard, 
                 isTraining: bool = True,
                 dropoutRate: float = 0.25) -> None:
        self.isTraining = isTraining
        self.dropoutRate = dropoutRate
        super().__init__(inputLayerSize, outputLayerSize, hasBias, weightInit)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.isTraining:
            return np.dot(x * np.random.binomial(1, 1 - self.dropoutRate, size=x.shape) / (1 - self.dropoutRate), self.weights)
        else:
            return np.dot(x, self.weights)



     


