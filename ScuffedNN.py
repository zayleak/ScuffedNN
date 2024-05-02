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
        return np.dot(net.W2, Sigmoid(np.dot(net.W1, x)))
    
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
        
        print(W2Outputs)

    # def forward(self, x):
    #     o1 = Linear(self.W1, x)
    #     act1 = Sigmoid(o1)
    #     o2 = Linear(self.W2, act1)
    #     act2 = Sigmoid(o2)
    #     o3 = Linear(self.W3, act2)

    #     self.W1Outputs.append(o1)
    #     self.W1Act.append(act1)
    #     self.W2Outputs.append(o2)
    #     self.W2Act.append(act2)
    #     self.W3Outputs.append(o3)
       
    #     self.yPreds.append(o3)

    #     return o3
    


    # def train(self, xTrain, yTrain):

    #     # xTrain list of [(inputLayerSize,)]
    #     # yTrain list of [(outputLayerSize,)]

    #     for x in xTrain:
    #         self.forward(x)

    #     print(self.W1Outputs)

    #     for index, y in enumerate(yTrain):
    #         deltaOne = self.yPreds[index] - y
    #         deltaTwos = derivSigmoid(self.W2Outputs[index])*np.dot(deltaOne, np.transpose(self.W3))

    #         deltaTwosShit = []
    #         for hiddenLayerNode in range(self.hiddenLayerSize):
    #             deltaTwosShit.append(deltaOne*self.W3*derivSigmoid(self.W2Act[index][hiddenLayerNode]))
            
    #         print(deltaTwosShit, deltaTwos)
    #         # deltaThrees = []
    #         # for inputLayerNode in range(self.inputLayerSize):
    #         #     deltaSum = 0
    #         #     for hiddenLayerNode in range(self.hiddenLayerSize):
    #         #         deltaSum += self.W2[inputLayerNode][hiddenLayerNode]*deltaTwos[hiddenLayerNode]
    #         #     deltaThrees.append(deltaSum*derivSigmoid(self.W1Act[index][hiddenLayerNode]))

    #         # #Layer one gradient
            
    #         # gradOne = deltaOne*self.W3

    #         # Layer two gradient

        
x = np.array([2, 1, 1])
net = SimpleFeedForward(3, 5, 3)
net.train(np.array([[20, 2, 3], [100, 200, 300]]), np.array([[1, 2, 3], [100, 200, 300]]), 1000)
# net.forward(x)
# print(net.W1Act, net.W2Act, net.W1Outputs, net.W2Outputs, net.yPreds)