from typing import Optional, List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ScuffedActivations
import ScuffedLayers
import ScuffedCostFunctions
import ScuffedWeightInit
from sklearn.datasets import load_iris
import ScuffedTrainUtil
import ScuffedRegularizations

class ScuffedNet():
    def __init__(self, 
                 costFunction : ScuffedCostFunctions.CostFunction = ScuffedCostFunctions.MSE(), 
                 regularizations: List[ScuffedRegularizations.RegularizationMethod] = [], 
                 alpha: float = 0.1,
                 batchSize: int = None) -> None:
        self.alpha = alpha
        self.costFunction = costFunction
        self.__layers = []
        self.regularizations = regularizations
        self.batchSize = batchSize

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

    def setDropoutLayers(self, setIsTraining: bool = True):
        for layer in self.__layers:
            if isinstance(layer["layer"], ScuffedLayers.DropoutLayer):
                layer["layer"].isTraining = setIsTraining

    def resetWeights(self):
        for layer in self.__layers:
            layer["layer"].initWeights()

    def getWeights(self):
        return [curLayer["layer"].weights for curLayer in self.__layers]

    def calculateTotalCost(self, YBatch: np.ndarray, finalTransform: np.ndarray, finalOutput: np.ndarray):
        totalCost = self.costFunction.computeCost(YBatch, finalTransform, finalOutput)
        sumWeightsSquared = sum([np.sum(np.square(weight)) for weight in self.getWeights()])
        for regularizationMethod in self.regularizations:
            totalCost += regularizationMethod.computeCost(sumWeightsSquared, YBatch.shape[0])
        return totalCost
    
    def updateLayerGradient(self, curLayer: ScuffedLayers.LinearLayer, prevOutputs: np.ndarray, curDelta: np.ndarray, numTraining: int):
        partialDerivs = prevOutputs[:, :, np.newaxis] * curDelta[:, np.newaxis, :]
        avgPartial = np.average(partialDerivs, axis=0)
        curLayer.backpropStep(-self.alpha, avgPartial, self.regularizations, numTraining)

    def epoch(self, XBatch: np.ndarray, YBatch: np.ndarray, epoch: int, printEpochs: bool = False):

        outputs, transforms = self.getForwardResults(XBatch)
        finalTransform = transforms[-1]
        numTraining = YBatch.shape[0]

        if printEpochs:
            print("Cost at epoch#{}: {}".format(epoch, self.calculateTotalCost(YBatch, finalTransform, outputs[-1])))

        # add final layers deltas/errors
        curDelta = self.costFunction.computeFirstDelta(outputs[-1], YBatch, self.__layers[-1]["activation"], finalTransform)
        self.updateLayerGradient(self.__layers[-1]["layer"], outputs[-2], curDelta, numTraining)

        for layerNum in reversed(range(len(self.__layers) - 1)):
            # curLayer + 1 because len(self.__layers) = len(outputs) - 1
            curOutputs = outputs[layerNum + 1]
            prevOutputs = outputs[layerNum]

            curTransforms = transforms[layerNum]
            nextLayer = self.__layers[layerNum + 1]
            curLayer = self.__layers[layerNum]
            nextWeights = nextLayer["layer"].weights

            removeWeightBias = nextWeights.T[:, 1:] if nextLayer["layer"].hasBias else nextWeights.T
            removeOutputBias = curOutputs[:, 1:] if nextLayer["layer"].hasBias else curOutputs
            curDelta = curLayer["activation"].derivActivation(removeOutputBias, curTransforms) * np.dot(curDelta, removeWeightBias)
            self.updateLayerGradient(curLayer["layer"], prevOutputs, curDelta, numTraining)

    def train(self, X: np.ndarray, Y: np.ndarray, numEpochs: int = 1000, printEpochs : bool = False):
        # xTrain list of [(inputLayerSize,)]
        # yTrain list of [(outputLayerSize,)]

        if len(self.__layers) == 0:
            return X
        
        self.setDropoutLayers(True)

        for epoch in range(numEpochs):
            everyHundred = (epoch % 100 == 0) & printEpochs
            
            if self.batchSize is None:
                self.epoch(X, Y, epoch, everyHundred)
                continue

            for batchStart in range(0, len(X), self.batchSize):
                batchEnd = batchStart + self.batchSize
                XBatch = X[batchStart:batchEnd]
                YBatch = Y[batchStart:batchEnd]
                self.epoch(XBatch, YBatch, epoch, everyHundred & (batchStart == 0))

        self.setDropoutLayers(False)

np.random.seed(48) 

regularizationsList = []

net = ScuffedNet(ScuffedCostFunctions.BinaryCrossEntropyLoss(), regularizationsList, 1)
net.addLayer(ScuffedLayers.LinearLayer(2, 5, True, ScuffedWeightInit.He), ScuffedActivations.Tanh())
net.addLayer(ScuffedLayers.LinearLayer(5, 3, True, ScuffedWeightInit.He), ScuffedActivations.LeakyReLU())
net.addLayer(ScuffedLayers.LinearLayer(3, 1, True, ScuffedWeightInit.He), ScuffedActivations.Sigmoid())

# net.train(np.array([[20, 2, 3], [100, 200, 300], [23, 23, 23]]), np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]]), 5000, printEpochs=True)
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

y = (y==2).astype('int')
# num_classes = len(np.unique(y))
# y_onehot = np.zeros((len(y), num_classes))
# y_onehot[np.arange(len(y)), y.flatten()] = 1

ScuffedTrainUtil.trainWithLearningCurve([70, 80, 90], [10000, 10000, 1000], net, X, y, False)
