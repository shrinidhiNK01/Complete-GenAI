"""
Variational Autoencoder (VAE) on MNIST in PyTorch.
This script implements:
1. Encoder: compresses 28x28 image to mean (mu) and log_variance (logvar).
2. Reparameterization Trick: samples z = mu + std * epsilon.
3. Decoder: reconstructs the image from the latent space z.
4. Loss function: Reconstruction Loss (BCE) + KL Divergence.

Run: python Complete_Course/GenAI/implementations/simple_vae.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# pyrefly: ignore [missing-import]
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Mean of latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space
        
        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Standard deviation std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # Sample random epsilon from standard normal distribution
        eps = torch.randn_like(std)
        # Return the sampled latent vector
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h))

    def forward(self, x):
        # Flatten image from (Batch, 1, 28, 28) to (Batch, 784)
        x_flat = x.view(-1, 784)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # 1. Reconstruction Loss: BCE between input pixels and output probabilities
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL Divergence: analytical form for Gaussian distributions
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VAE on device: {device}")
    
    # Load MNIST
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    os.makedirs('Complete_Course/GenAI/results', exist_ok=True)
    
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch} - Average Loss: {train_loss / len(train_loader.dataset):.4f}")
        
        # Save reconstructed images to inspect quality
        with torch.no_grad():
            # Generate new samples by feeding random vectors to the decoder
            sample = torch.randn(64, 20).to(device)
            sample_out = model.decode(sample).cpu()
            save_image(sample_out.view(64, 1, 28, 28), f'Complete_Course/GenAI/results/sample_{epoch}.png')
            
    print("Training finished! Generated images saved to 'Complete_Course/GenAI/results/'")

if __name__ == '__main__':
    main()
