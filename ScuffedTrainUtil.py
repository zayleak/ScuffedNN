import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np
from sklearn.datasets import load_iris

def getAccuracies(XSet: np.ndarray, YSet: np.ndarray, net) -> float:
    size = len(XSet)
    trainPredictions = net.forward(XSet)[-1]
    trainProbs = [1 if XVal > 0.5 else 0 for XVal in trainPredictions]
    trainCorrectPredictions = [YVal == XVal for XVal, YVal in zip(trainProbs, YSet)]
    trainAccuracy = sum(trainCorrectPredictions) / size
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

        for layer in net.getLayers():
            layer["layer"].initWeights()

    plt.plot(trainSizes, trainAccuracies, label='Training Accuracy')
    plt.plot(trainSizes, validationAccuracies, label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()