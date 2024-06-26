from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import ScuffedActivations
import ScuffedTrainUtil

class CostFunction(ABC):

    @abstractmethod
    def computeFirstDelta(self, finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def computeCost(self, yTrain: np.ndarray, finalTransform: np.ndarray, finalOutput: np.ndarray) -> float:
        pass


class MSE(CostFunction):

    def computeFirstDelta(self, finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        return (finalOutput - yTrain)*activationFcn.derivActivation(finalOutput, finalTransform)
    
    def computeCost(self, yTrain: np.ndarray, finalTransform: np.ndarray, finalOutput: np.ndarray) -> float:
        m = yTrain.shape[0]
        return (1/m)*np.sum(np.square(finalOutput - yTrain))
    
# this should be used if the final layers activation function is a sigmoid function saying if its part of a certian class or not
class BinaryCrossEntropyLoss(CostFunction):

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def computeFirstDelta(self, finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        m = yTrain.shape[0]
        return (1/m)*((1/(1+np.exp(-finalTransform))) - yTrain)
    
    def computeCost(self, yTrain: np.ndarray, finalTransform: np.ndarray, finalOutput: np.ndarray):
        m = yTrain.shape[0]
        return (1/m) * np.sum(np.maximum(finalTransform, 0) - finalTransform*yTrain + np.log(1+ np.exp(- np.abs(finalTransform))))
    
    def getPreds(self, trainPredictions) -> List:
        return [1 if XVal > self.threshold else 0 for XVal in trainPredictions]

    def getCorrectPreds(self, trainPredictions: np.ndarray, YSet: np.ndarray) -> List:
        preds = self.getPreds(trainPredictions)
        return [YVal == XVal for XVal, YVal in zip(preds, YSet)]
    
# this should be used if the final layers activation function is a softmax function for probabilities
class MultiCrossEntropyLoss(CostFunction):

    def computeFirstDelta(self, finalOutput: np.ndarray, yTrain: np.ndarray, activationFcn: ScuffedActivations.ActivationFunction, finalTransform: np.ndarray) -> np.ndarray:
        m = yTrain.shape[0]
        return (1/m)*((1/(1+np.exp(-finalTransform))) - yTrain)
    
    def computeCost(self, yTrain: np.ndarray, finalTransform: np.ndarray, finalOutput: np.ndarray) -> float:
        m = yTrain.shape[0]
        logProb = np.log(finalOutput) * yTrain
        return -np.sum(logProb)*(1/m)
    
    def getPreds(self, trainPredictions) -> List:
        return ScuffedTrainUtil.oneHotEncode(trainPredictions)
    
    def getCorrectPreds(self, trainPredictions: np.ndarray, YSet: np.ndarray) -> List:
        preds = self.getPreds(trainPredictions)
        return [np.all(YVal == XVal) for XVal, YVal in zip(preds, YSet)]
    
