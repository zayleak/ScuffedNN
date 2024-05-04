import torch

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, 3)
        self.sigmoid = nn.Sigmoid()
        
        self.fc1.weight.data.uniform_(-1, 1)
        self.fc2.weight.data.uniform_(-1, 1)
        self.fc3.weight.data.uniform_(-1, 1)
        self.fc4.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
# Create an instance of the neural network
    
net = Net()

# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Convert the input and target data to tensors
inputs = torch.tensor([[20, 2, 3], [100, 200, 300], [23, 23, 23]], dtype=torch.float32)
targets = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.float32)

# Train the neural network
for epoch in range(500):
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

# Make a prediction
prediction = net(torch.tensor([100, 200, 300], dtype=torch.float32))
print(prediction)



