"""
Multi-Layer Perceptron (MLP) classification using PyTorch.
This script demonstrates the standard PyTorch workflow for loading data,
defining a network, training it, and evaluating accuracy.

Run: python Complete_Course/DL/implementations/mlp_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed
torch.manual_seed(42)

# 1. Define Model Architecture
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PyTorchMLP, self).__init__()
        # Linear layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Linear layer 2
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # Note: In PyTorch, nn.CrossEntropyLoss expects raw logits, 
        # so we do NOT apply softmax in the forward pass.
        return out


def main():
    # 2. Generate and scale synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. Instantiate Model, Loss, and Optimizer
    input_size = 20
    hidden_size = 64
    num_classes = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PyTorchMLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 5. Training Loop
    num_epochs = 20
    print("\n--- Training PyTorch Model ---")
    for epoch in range(num_epochs):
        model.train()  # set model to training mode
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Clear past gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass (calculate gradients)
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

    # 6. Evaluation Loop
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # disable gradient tracking for faster inference
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)  # get indices of max logit
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f"\nTest Accuracy on 200 samples: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
