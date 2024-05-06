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

def getPrecisionRecall(curClass: np.ndarray, curYClass: np.ndarray):
    truePositives = sum((curClass == 1) & (curYClass == 1))
    falsePositives = sum((curClass == 1) & (curYClass == 0))
    falseNegitives = sum((curClass == 0) & (curYClass == 1))

    if truePositives == 0:
        return 0.0, 0.0

    return truePositives / (truePositives + falsePositives), truePositives / (truePositives + falseNegitives)

def getFScore(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0.0
    else:
        return precision * recall / (precision + recall)

def nanZero(number: int) -> float:
    if np.isnan(number):
        return 0.0
    else:
        return number

def printPerformenceMetrics(XSet: np.ndarray, YSet: np.ndarray, net) -> float:
    outputs, _ = net.getForwardResults(np.array(XSet))
    predictions = net.costFunction.getPreds(outputs[-1])

    if isinstance(predictions[0], int):
        predictions = np.array(predictions)[:, np.newaxis]

    for classNumber in range(YSet.shape[1]):
        curClass = predictions[:, classNumber]
        curYClass = YSet[:, classNumber]
        precision, recall = getPrecisionRecall(curClass, curYClass)
        precision = nanZero(precision)
        recall = nanZero(recall)

        print(f"Precision for classNumber#{classNumber}: {precision}")
        print(f"Recall for classNumber#{classNumber}: {recall}")
        print(f"F-Score for classNumber#{classNumber}: {getFScore(precision, recall)}")
        


def trainWithLearningCurve(trainSizes: List[int], epochList: List[int], net, Xall: np.ndarray, Yall: np.ndarray, printEpochs: bool = True) -> None:
    trainAccuracies = []
    validationAccuracies = []

    for curTrainIndex, (size, numEpochs) in enumerate(zip(trainSizes, epochList)):
        XTrain = Xall[:size]
        YTrain = Yall[:size]
        XVal = Xall[size:]
        YVal = Yall[size:]

        net.train(XTrain, YTrain, numEpochs=numEpochs, printEpochs=printEpochs)

        trainAccuracies.append(getAccuracies(XTrain, YTrain, net))
        validationAccuracies.append(getAccuracies(XVal, YVal, net))
        print("-----------------------------")
        print(f"Performance metrics for training index#{curTrainIndex} (On the validation set)")
        printPerformenceMetrics(XVal, YVal, net)
        print("-----------------------------")
        net.resetWeights()

    plt.plot(trainSizes, trainAccuracies, label='Training Accuracy')
    plt.plot(trainSizes, validationAccuracies, label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()