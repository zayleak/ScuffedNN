from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class RegularizationMethod(ABC):

    @abstractmethod
    def computeCost(self, sumWeightSquared: float, numTraining: int) -> np.ndarray:
        pass

    @abstractmethod
    def computeGradientCost(self, weight: np.ndarray, numTraining: int) -> np.ndarray:
        pass

class L1Regularization(RegularizationMethod):

    def __init__(self, regLambda: float = 0.01) -> None:
        self.regLambda = regLambda

    def computeCost(self, sumWeightSquared: float, numTraining: int) -> float:
        return (1/numTraining)*(self.regLambda/2)*sumWeightSquared
        
    def computeGradientCost(self, weight: np.ndarray, numTraining: int) -> np.ndarray:
        return (self.regLambda/numTraining)*weight
    


