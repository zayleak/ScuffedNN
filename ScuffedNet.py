from typing import Optional, List

import numpy as np
import ScuffedActivations
import ScuffedLayers

class ScuffedNet():
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.__layers = []

    def addLayer(self, layer: ScuffedLayers.LinearLayer, activationFcn: ScuffedActivations.ActivationFunction) -> None:
        self.__layers.append({"layer": layer, "activation": activationFcn})

    def forward(self, xTrain: np.ndarray):
        outputs = []
        output = xTrain
        for nextLayer in range(len(self.__layers)):
            nextLayer = self.__layers[nextLayer]
            output = np.hstack((np.ones((len(xTrain), 1)), output)) if nextLayer["layer"].hasBias else output
            outputs.append(output)
            output = nextLayer["layer"].forward(output)
            output = nextLayer["activation"].activation(output)
        outputs.append(output)
        return outputs
    
    def makePred(self, x: List[float]):
        return self.forward(np.array([x]))[-1].flatten()

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, numIters: int = 1000) -> np.ndarray:
        # xTrain list of [(inputLayerSize,)]
        # yTrain list of [(outputLayerSize,)]

        if len(self.__layers) <= 1:
            return xTrain

        for _ in range(numIters):
            
            deltas = []
            outputs = self.forward(xTrain)
            
            # add final layers deltas/errors
            deltas.append(outputs[-1] - yTrain)

            for curLayer in reversed(range(len(self.__layers) - 1)):
                # curLayer + 1 because len(self.__layers) = len(outputs) - 1
                curOutputs = outputs[curLayer + 1]

                nextLayer = self.__layers[curLayer + 1]
                curLayer = self.__layers[curLayer]

                nextWeights = nextLayer["layer"].weights
                
                removeWeightBias = nextWeights.T[:, 1:] if nextLayer["layer"].hasBias else nextWeights.T
                removeOutputBias = curOutputs[:, 1:] if curLayer["layer"].hasBias else curOutputs
                
                deltas.append(curLayer["activation"].derivActivation(removeOutputBias) * np.dot(deltas[-1], removeWeightBias))

            for curOutput, curDelta, curLayer in zip(outputs[:-1], reversed(deltas), self.__layers):
                curPartial = curDelta[:, np.newaxis, :] * curOutput[: , :, np.newaxis]
                avgPartial = np.average(curPartial, axis=0)
                curLayer["layer"].gradientUpdate(-self.alpha, avgPartial)
            

net = ScuffedNet()
net.addLayer(ScuffedLayers.LinearLayer(3, 5), ScuffedActivations.Sigmoid)
net.addLayer(ScuffedLayers.LinearLayer(5, 3), ScuffedActivations.Identity)

net.train(np.array([[20, 2, 3], [100, 200, 300]]), np.array([[1, 2, 3], [100, 200, 300]]), 20)
print(net.makePred([20, 2, 3]))


    