from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import ScuffedActivations

class CostFunction(ABC):

    @abstractmethod
    def computeFirstDelta(finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def computeCost(yTrain: np.ndarray, finalTransform: np.ndarray):
        pass

class MSE(CostFunction):

    def computeFirstDelta(finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        return (finalOutput - yTrain)*activationFcn.derivActivation(finalOutput)
    
    def computeCost(yTrain: np.ndarray, finalTransform: np.ndarray):
        return super().computeCost(finalTransform)
    
# this should be used if the final layers activation function is a softmax function for probabilities
class BinaryCrossEntropyLoss(CostFunction):

    def computeFirstDelta(finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        m = yTrain.shape[0]
        return (1/m)*((1/(1+np.exp(-finalTransform))) - yTrain)
    
    def computeCost(yTrain: np.ndarray, finalTransform: np.ndarray):
        m = yTrain.shape[0]
        return (1/m) * np.sum(np.maximum(finalTransform, 0) - finalTransform*yTrain + np.log(1+ np.exp(- np.abs(finalTransform))))
    
# this should be used if the final layers activation function is a sigmoid function saying if its part of a certian class or not
class MultiCrossEntropyLoss(CostFunction):

    def computeFirstDelta(finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        return (finalOutput - yTrain)
    
    def computeCost(yTrain: np.ndarray, finalTransform: np.ndarray):
        return super().computeCost(finalTransform)
