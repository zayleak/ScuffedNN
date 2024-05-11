## Overview:
ScuffedNN is a self made neural net using various different concepts across ML. It was made with primarly numpy, a little bit of matplot lib (for the analysis functions), and no tensorflow/pytorch. 
I primarly used OOP for all of the functionality.

## Features:
DropoutLayers
L1 Regularization
Multiple layers
MSE, Binary Cross Entropy, and Multi-class Cross Entropy loss functions
Tanh, ReLU, LeakyReLU, ELU, Sigmoid, Softmax Activation functions
batch gradient descent
Learning curves
Random initialization, Xavier initialization, He initialization
Accuracy, Precision, Recall, and F1 Score

## Example Usage:
In this example:
I created an L1 Regularization with a lambda of 0.01 (See https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c for more details)
I defined the net with Binary Cross Entropy Loss, A regularizationList (In our case just L1 Regularization), and a learning rate of 1
I then I created 3 Layers with:

Layer 1: Input size 2 output size 5 linear layer, with bias (The true), He initialization, and Tanh activation
Layer 2: Input size 5 output size 3 linear layer, with bias (The true), He initialization, and LeakyReLU activation
Layer 3: Input size 3 output size 1 linear layer, with bias (The true), He initialization, and Sigmoid activation (due to binary cross entropy loss)

![image](https://github.com/zayleak/ScuffedNN/assets/90633128/2479b715-57d9-4072-8e74-a7c4e64b7eb9)

