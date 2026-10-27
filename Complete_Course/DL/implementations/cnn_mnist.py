"""
Convolutional Neural Network (CNN) on MNIST in PyTorch.
This file explains how spatial dimensions change at each layer of a CNN:
1. Input: (Batch, 1, 28, 28)
2. Conv1 (kernel=3, stride=1, padding=1): Out channels = 16. Size = (28 - 3 + 2(1))/1 + 1 = 28. Output: (Batch, 16, 28, 28)
3. MaxPool1 (kernel=2, stride=2): Size = 28/2 = 14. Output: (Batch, 16, 14, 14)
4. Conv2 (kernel=3, stride=1, padding=1): Out channels = 32. Size = (14 - 3 + 2(1))/1 + 1 = 14. Output: (Batch, 32, 14, 14)
5. MaxPool2 (kernel=2, stride=2): Size = 14/2 = 7. Output: (Batch, 32, 7, 7)
6. Flatten: 32 * 7 * 7 = 1568 features.
7. Fully Connected layers mapping to 128 then to 10 classes.

Run: python Complete_Course/DL/implementations/cnn_mnist.py --epochs 2
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions again
        
        self.flatten = nn.Flatten()
        
        # Output of pool2 is (Batch, 32, 7, 7), so flattened features = 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # 2. Prepare MNIST data with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST mean and standard deviation
    ])
    
    # Download dataset
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    # 3. Instantiate model, loss, and optimizer
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(xb)}/{len(train_loader.dataset)}] - Batch Loss: {loss.item():.4f}")
                
        print(f"-> Epoch {epoch} Complete. Average Train Loss: {total_loss / len(train_loader):.4f}\n")

    # 5. Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            
    print(f"Final Test Accuracy: {correct/total * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    train(args)
