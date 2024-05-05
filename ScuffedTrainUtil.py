import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np

def oneHotEncode(data: np.ndarray) -> np.ndarray:
    numUnique = data.shape[1]
    oneHot = np.zeros((len(data), numUnique))
    for i, val in enumerate(data):
        oneHot[i, np.argmax(val)] = 1
    return oneHot

def getAccuracies(XSet: np.ndarray, YSet: np.ndarray, net) -> float:
    size = len(XSet)
    outputs, _ = net.getForwardResults(np.array(XSet))
    predictions = net.costFunction.getCorrectPreds(outputs[-1], YSet)
    trainAccuracy = sum(predictions) / size
    return trainAccuracy

def trainWithLearningCurve(trainSizes: List[int], epochList: List[int], net, Xall: np.ndarray, Yall: np.ndarray, printEpochs: bool = True) -> None:
    trainAccuracies = []
    validationAccuracies = []

    for size, numEpochs in zip(trainSizes, epochList):
        XTrain = Xall[:size]
        YTrain = Yall[:size]
        XVal = Xall[size:]
        YVal = Yall[size:]

        net.train(XTrain, YTrain, numEpochs=numEpochs, printEpochs=printEpochs)

        trainAccuracies.append(getAccuracies(XTrain, YTrain, net))
        validationAccuracies.append(getAccuracies(XVal, YVal, net))

        net.resetWeights()

    plt.plot(trainSizes, trainAccuracies, label='Training Accuracy')
    plt.plot(trainSizes, validationAccuracies, label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()