"""
Logistic Regression from Scratch (Sigmoid, Cross-Entropy Loss) and Scikit-Learn.
This script demonstrates classification using logistic regression.

Run: python Complete_Course/ML/implementations/logistic_regression.py
"""

import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Set random seed
np.random.seed(42)

class ScratchLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # Sigmoid activation squashes z to a value between 0 and 1
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip to avoid overflow

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # 1. Forward pass: compute linear combo and pass through sigmoid
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # 2. Compute Binary Cross-Entropy Loss (Log Loss)
            # Add a small epsilon to log to avoid log(0)
            eps = 1e-15
            loss = -np.mean(y * np.log(y_predicted + eps) + (1 - y) * np.log(1 - y_predicted + eps))
            self.loss_history.append(loss)
            
            # 3. Compute gradients (derivatives of log loss w.r.t weights & bias)
            # dw = (1/N) * sum( X * (y_predicted - y) )
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/N) * sum( y_predicted - y )
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if epoch % 100 == 0:
                print(f"Scratch Model - Epoch {epoch}/{self.epochs}, Log Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)


def main():
    # 1. Generate classification dataset
    X, y = make_classification(
        n_samples=200, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2, 
        random_state=42, 
        n_clusters_per_class=1
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Training Scratch Logistic Regression ---")
    scratch_model = ScratchLogisticRegression(learning_rate=0.1, epochs=1000)
    scratch_model.fit(X_train, y_train)
    scratch_preds = scratch_model.predict(X_test)

    print("\n--- Training Scikit-Learn Logistic Regression ---")
    sklearn_model = LogisticRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)

    # Compare accuracies
    scratch_acc = accuracy_score(y_test, scratch_preds)
    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    
    print("\n--- Model Performance Comparison ---")
    print(f"Scratch Accuracy:    {scratch_acc * 100:.2f}%")
    print(f"Sklearn Accuracy:    {sklearn_acc * 100:.2f}%")
    print("\nScratch Confusion Matrix:")
    print(confusion_matrix(y_test, scratch_preds))

    # Plot the decision boundary
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Decision boundary for scratch model
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', alpha=0.8)
    
    # Calculate decision boundary line: w0*x0 + w1*x1 + b = 0 => x1 = -(w0*x0 + b) / w1
    x_values = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 100)
    w = scratch_model.weights
    b = scratch_model.bias
    y_values = -(w[0] * x_values + b) / w[1]
    
    plt.plot(x_values, y_values, label='Decision Boundary', color='black', linewidth=2)
    plt.title("Scratch Logistic Regression Boundary (Test Set)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # Plot 2: Loss History
    plt.subplot(1, 2, 2)
    plt.plot(scratch_model.loss_history, color='orange')
    plt.title("Scratch Model Cross-Entropy Loss History")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")

    plt.tight_layout()
    plt.savefig("Complete_Course/ML/logistic_regression_results.png")
    print("\nPlot saved successfully as 'Complete_Course/ML/logistic_regression_results.png'")

if __name__ == '__main__':
    main()
