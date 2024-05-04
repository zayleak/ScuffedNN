import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load the iris dataset
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

# Convert the input and target data to tensors
inputs = torch.tensor(X, dtype=torch.float32)
targets = torch.tensor(y, dtype=torch.long)

# Fit and transform y to one-hot encoded vectors
encoder = OneHotEncoder(categories='auto')
y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()
targets_one_hot = torch.tensor(y_one_hot, dtype=torch.float32)

# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1)

# Train the neural network
for epoch in range(10000):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = net(inputs)

    # Calculate the loss
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

# Make predictions on new data
new_data = inputs
predictions = net(new_data)
max_probabilities, predicted_labels = torch.max(predictions, dim=1)
correct_predictions = (predicted_labels == targets).tolist()
print(sum(correct_predictions))
