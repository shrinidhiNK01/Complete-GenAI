"""
Linear Regression from Scratch (Gradient Descent) and Scikit-Learn.
This script demonstrates the exact math and step-by-step implementation of simple linear regression.

Run: python Complete_Course/ML/implementations/linear_regression.py
"""

import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

class ScratchLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        # X shape: (n_samples, n_features)
        # y shape: (n_samples,)
        n_samples, n_features = X.shape
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Gradient Descent Loop
        for epoch in range(self.epochs):
            # Forward pass: predict y_hat = X * w + b
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute Mean Squared Error Loss
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients (partial derivatives of loss w.r.t weights and bias)
            # dw = (2/N) * sum( X * (y_predicted - y) )
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (2/N) * sum( y_predicted - y )
            db = (2 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if epoch % 100 == 0:
                print(f"Scratch Model - Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


def main():
    # 1. Generate synthetic data: y = 3.5 * x + 4.0 + noise
    X = np.random.rand(100, 1) * 10  # 100 random points between 0 and 10
    true_weights = 3.5
    true_bias = 4.0
    noise = np.random.randn(100, 1) * 2  # Gaussian noise
    y = (true_weights * X + true_bias + noise).squeeze()

    print("--- Training Scratch Linear Regression ---")
    scratch_model = ScratchLinearRegression(learning_rate=0.01, epochs=1000)
    scratch_model.fit(X, y)
    scratch_preds = scratch_model.predict(X)

    print("\n--- Training Scikit-Learn Linear Regression ---")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_preds = sklearn_model.predict(X)

    # Compare parameters
    print("\n--- Parameter Comparison ---")
    print(f"True Relation:       y = {true_weights} * x + {true_bias}")
    print(f"Scratch Estimates:   y = {scratch_model.weights[0]:.4f} * x + {scratch_model.bias:.4f}")
    print(f"Sklearn Estimates:   y = {sklearn_model.coef_[0]:.4f} * x + {sklearn_model.intercept_:.4f}")

    # Compare MSE
    scratch_mse = mean_squared_error(y, scratch_preds)
    sklearn_mse = mean_squared_error(y, sklearn_preds)
    print(f"Scratch MSE:         {scratch_mse:.4f}")
    print(f"Sklearn MSE:         {sklearn_mse:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Regression Lines
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data points')
    plt.plot(X, scratch_preds, color='red', linewidth=3, label='Scratch (Gradient Descent)')
    plt.plot(X, sklearn_preds, color='green', linestyle='--', linewidth=2, label='Sklearn (Normal Equation)')
    plt.title("Linear Regression Fit Comparison")
    plt.xlabel("X (Feature)")
    plt.ylabel("y (Label)")
    plt.legend()
    
    # Plot 2: Loss History for Scratch Model
    plt.subplot(1, 2, 2)
    plt.plot(scratch_model.loss_history, color='purple')
    plt.title("Scratch Model Gradient Descent Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE) Loss")
    
    plt.tight_layout()
    # Save the plot to the ML folder for inspection
    plt.savefig("Complete_Course/ML/linear_regression_results.png")
    print("\nPlot saved successfully as 'Complete_Course/ML/linear_regression_results.png'")
    # Note: plt.show() might block, so we save it as a file instead.

if __name__ == '__main__':
    main()
