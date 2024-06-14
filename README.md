## Overview
ScuffedNN is a self made neural net using various different concepts across ML. It was made with primarly numpy, a little bit of matplot lib (for the analysis functions), and no tensorflow/pytorch. 
I primarly used OOP for all of the functionality.

I have also wrote some articles documenting my experiences in learning about neural nets: 

Article 1: https://zayleak.substack.com/p/creating-a-neural-net-from-scratch

Article 2: https://zayleak.substack.com/p/creating-a-neural-net-from-scratch-cdf

Article 3: https://zayleak.substack.com/p/creating-a-neural-net-from-scratch-cf9

## Features
DropoutLayers

L1 Regularization

Multiple layers

MSE, Binary Cross Entropy, and Multi-class Cross Entropy loss functions

Tanh, ReLU, LeakyReLU, ELU, Sigmoid, Softmax Activation functions

batch gradient descent

Learning curves

Random initialization, Xavier initialization, He initialization

Accuracy, Precision, Recall, and F1 Score

## Example Usage
In this example:

I created an L1 Regularization with a lambda of 0.01 (See https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c for more details)

I defined the net with Binary Cross Entropy Loss, A regularizationList (In our case just L1 Regularization), and a learning rate of 1

I then I created 3 Layers with:

Layer 1: Input size 2 output size 5 linear layer, with bias (The true), He initialization, and Tanh activation

Layer 2: Input size 5 output size 3 linear layer, with bias (The true), He initialization, and LeakyReLU activation

Layer 3: Input size 3 output size 1 linear layer, with bias (The true), He initialization, and Sigmoid activation (due to binary cross entropy loss)

![image](https://github.com/zayleak/ScuffedNN/assets/90633128/2479b715-57d9-4072-8e74-a7c4e64b7eb9)

## Training Example 

In this example:

Using the same network as above

did data preprocessing for Iris flowers (See https://github.com/RafayAK/NothingButNumPy/blob/master/Understanding_and_Creating_Binary_Classification_NNs/2_layer_toy_neural_network_on_all_iris_data.ipynb for more info)

Then called trainWithLearningCurve util with Train sizes 70, 80, 90 and epoch sizes 10000, 10000, and 1000 respectively with my current neural net. The false is if you want to print epoch loss data (In this case I did not).

Also, note you input the full data amount which includes both the validation and training data. Therefore the validation set sizes will be 150 - 70 = 80, 150 - 80 = 70, and 150 - 90 = 60 size respectively

![image](https://github.com/zayleak/ScuffedNN/assets/90633128/704271be-91eb-4095-87ca-02df8fd2a2c1)

This is the output I got for my model

![image](https://github.com/zayleak/ScuffedNN/assets/90633128/583659ad-4dcd-426b-bbd4-a90157f422d1)


