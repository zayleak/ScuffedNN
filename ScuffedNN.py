import numpy as np

def Sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def derivSigmoid(vector):
    sig = Sigmoid(vector)
    return sig * (1 - sig)

class SimpleFeedForward:

    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, alpha=0.1):
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        self.alpha = alpha

        self.W1 = np.ones((inputLayerSize + 1, hiddenLayerSize))
        self.W2 = np.ones((hiddenLayerSize + 1, outputLayerSize))
        
    def makePred(self, x):
        inputWithBias = np.hstack((np.ones((1, 1)), np.array([x])))
        W1Output = np.hstack((np.ones((1, 1)), Sigmoid(np.dot(inputWithBias, self.W1))))
        return np.dot(W1Output, self.W2).flatten()
    
    def train(self, xTrain, yTrain, numIters=1000):

        #     # xTrain list of [(inputLayerSize,)]
        #     # yTrain list of [(outputLayerSize,)]

        for _ in range(numIters):

            inputWithBias = np.hstack((np.ones((len(xTrain), 1)), xTrain))
            W1Outputs = np.hstack((np.ones((len(xTrain), 1)), Sigmoid(np.dot(inputWithBias, self.W1))))
            W2Outputs = np.dot(W1Outputs, self.W2)

            deltaThrees = W2Outputs - yTrain

            deltaTwos = derivSigmoid(W1Outputs[:, 1:]) * np.dot(deltaThrees, self.W2.T[:, 1:])
            # deltaTwos size (N x O) -> (N x 1 x O) and inputWith bias -> (N x F + 1 x 1)
            # resulting in collections of matrices of partial derivs(w, i, j) -> wth training vector, oi*deltaj       
            W1Partial = deltaTwos[:, np.newaxis, :] * inputWithBias[:, :, np.newaxis]
            W2Partial = deltaThrees[:, np.newaxis, :] * W1Outputs[:, :, np.newaxis] 

            # average across axis (w, i, j) w axis -> matrix will be size (F + 1 x O) 
            avgW1Partial = np.average(W1Partial, axis=0)
            avgW2Partial = np.average(W2Partial, axis=0)
            self.W1 += -self.alpha * avgW1Partial
            self.W2 += -self.alpha * avgW2Partial
      

x = np.array([2, 1, 1])
net = SimpleFeedForward(3, 5, 3)
net.train(np.array([[20, 2, 3], [100, 200, 300]]), np.array([[1, 2, 3], [100, 200, 300]]), 20)

