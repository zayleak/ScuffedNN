from typing import Optional, List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ScuffedActivations
import ScuffedLayers
import ScuffedCostFunctions
import ScuffedWeightInit
from sklearn.datasets import load_iris
import ScuffedTrainUtil

class ScuffedNet():
    def __init__(self, costFunction : ScuffedCostFunctions.CostFunction = ScuffedCostFunctions.MSE, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.costFunction = costFunction
        self.__layers = []

    def addLayer(self, layer: ScuffedLayers.LinearLayer, activationFcn: ScuffedActivations.ActivationFunction) -> None:
        self.__layers.append({"layer": layer, "activation": activationFcn})
    
    def getLayers(self):
        return self.__layers
    
    def getForwardResults(self, xTrain: np.ndarray):
        outputs = []
        transforms = []
        output = xTrain
        for nextLayer in range(len(self.__layers)):
            nextLayer = self.__layers[nextLayer]
            output = np.hstack((np.ones((len(xTrain), 1)), output)) if nextLayer["layer"].hasBias else output
            outputs.append(output)
            output = nextLayer["layer"].forward(output)
            transforms.append(output)
            output = nextLayer["activation"].activation(output)
        outputs.append(output)
        return outputs, transforms
    
    def makePred(self, x: List[float]):
        outputs, _ = self.getForwardResults(np.array([x]))
        return outputs[-1].flatten()
    
    def resetWeights(self):
        for layer in self.__layers:
            layer["layer"].initWeights()

    def train(self, xTrain: np.ndarray, yTrain: np.ndarray, numEpochs: int = 1000, printEpochs : bool = False) -> None:
        # xTrain list of [(inputLayerSize,)]
        # yTrain list of [(outputLayerSize,)]

        if len(self.__layers) == 0:
            return xTrain

        for epoch in range(numEpochs):
            
            deltas = []
            outputs, transforms = self.getForwardResults(xTrain)
            finalTransform = transforms[-1]
            
            if ((epoch % 100) == 0 or numEpochs - 1 == epoch) and printEpochs:
                print("Cost at epoch#{}: {}".format(epoch, self.costFunction.computeCost(yTrain, finalTransform, outputs[-1])))

            # add final layers deltas/errors
            deltas.append(self.costFunction.computeFirstDelta(outputs[-1], yTrain, self.__layers[-1]["activation"], finalTransform))

            for curLayer in reversed(range(len(self.__layers) - 1)):
                # curLayer + 1 because len(self.__layers) = len(outputs) - 1
                curOutputs = outputs[curLayer + 1]
                curTransforms = transforms[curLayer]
                nextLayer = self.__layers[curLayer + 1]
                curLayer = self.__layers[curLayer]
                
                nextWeights = nextLayer["layer"].weights

                removeWeightBias = nextWeights.T[:, 1:] if nextLayer["layer"].hasBias else nextWeights.T
                removeOutputBias = curOutputs[:, 1:] if nextLayer["layer"].hasBias else curOutputs
                deltas.append(curLayer["activation"].derivActivation(removeOutputBias, curTransforms) * np.dot(deltas[-1], removeWeightBias))
            
            for curOutput, curDelta, curLayer in zip(outputs[:-1], reversed(deltas), self.__layers):
                curPartial = curOutput[:, :, np.newaxis] * curDelta[:, np.newaxis, :]
                avgPartial = np.average(curPartial, axis=0)
                curLayer["layer"].gradientUpdate(-self.alpha, avgPartial)

np.random.seed(48) 

net = ScuffedNet(ScuffedCostFunctions.MultiCrossEntropyLoss(), 1)
net.addLayer(ScuffedLayers.LinearLayer(2, 5, True, ScuffedWeightInit.Xavier), ScuffedActivations.LeakyReLU())
net.addLayer(ScuffedLayers.LinearLayer(5, 3, True, ScuffedWeightInit.Xavier), ScuffedActivations.Tanh())
net.addLayer(ScuffedLayers.LinearLayer(3, 3, True, ScuffedWeightInit.Xavier), ScuffedActivations.Sigmoid())




# net.train(np.array([[20, 2, 3], [100, 200, 300], [23, 23, 23]]), np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]]), 5000)
# print(net.makePred([100, 200, 300]))


iris = load_iris()

X = iris.data
X = iris.data[:, 2:] 
y = iris.target
# Restructure y to have shape (150, 1)
y = y.reshape(-1, 1)

# One-hot encode y



 # for reproducible randomization 
random_indices = np.random.permutation(len(X))  # genrate random permutation of indices

X= X[random_indices]
y = y[random_indices]

# y = (y==2).astype('int')
num_classes = len(np.unique(y))
y_onehot = np.zeros((len(y), num_classes))
y_onehot[np.arange(len(y)), y.flatten()] = 1

ScuffedTrainUtil.trainWithLearningCurve([70, 80, 90], [1000, 1000, 1000], net, X, y_onehot)
