import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def predict(model, input_data):
    # Convert input data to PyTorch tensor
    input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Forward pass to get predictions
    with torch.no_grad():
        predictions = model(input_data)
        
    return predictions.numpy()

class SimpleFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.fc1.weight.data.fill_(1)
        self.fc2.weight.data.fill_(1)
        
    def forward(self, x):
        # Input layer
        x = torch.sigmoid(self.fc1(x))
        # Hidden layer with sigmoid activation
        x = self.fc2(x)
        return x

def train(input_data, target_data, num_steps, learning_rate):
    # Convert numpy arrays to PyTorch tensors
    input_data = torch.tensor(input_data, dtype=torch.float32)
    target_data = torch.tensor(target_data, dtype=torch.float32)
    
    # Define model, loss function, and optimizer
    model = SimpleFeedForward(input_size=3, hidden_size=5, output_size=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for step in range(num_steps):
        # Forward pass
        outputs = model(input_data)
        
        # Compute loss
        loss = criterion(outputs, target_data)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step [{step + 1}/{num_steps}], Loss: {loss.item():.4f}")

    print(predict(model, np.array([[20, 2, 3], [100, 200, 300]])))

# Define input, target, and training parameters
input_data = np.array([[20, 2, 3], [100, 200, 300]])
target_data = np.array([[1, 2, 3], [100, 200, 300]])
num_steps = 10000
learning_rate = 0.1

# Train the model
train(input_data, target_data, num_steps, learning_rate)


