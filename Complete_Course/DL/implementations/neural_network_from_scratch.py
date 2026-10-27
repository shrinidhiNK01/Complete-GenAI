"""
Multi-Layer Perceptron (Neural Network) from Scratch using ONLY NumPy.
This script demonstrates the exact mathematical backward pass (backpropagation)
for a 2-layer neural network (Input -> Hidden -> Output).

Run: python Complete_Course/DL/implementations/neural_network_from_scratch.py
"""

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

class ScratchNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        
        # Initialize weights and biases
        # He/Xavier-like initialization for weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        # a is the output of the sigmoid: sigmoid(z)
        # Derivative of sigmoid(z) is sigmoid(z) * (1 - sigmoid(z))
        return a * (1.0 - a)

    def _softmax(self, z):
        # Softmax along row axis
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Shift for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # 1. Input to Hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)  # Hidden activation
        
        # 2. Hidden to Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._softmax(self.z2)  # Output probabilities
        return self.a2

    def backward(self, X, y_one_hot):
        n_samples = X.shape[0]

        # 1. Output error / gradient
        # Derivative of Cross-Entropy w.r.t Softmax input (z2) is simply: (prediction - true)
        dZ2 = self.a2 - y_one_hot  # shape: (n_samples, output_size)
        
        # Gradients for W2 and b2
        dW2 = np.dot(self.a1.T, dZ2) / n_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

        # 2. Backpropagate error to Hidden layer
        # dZ1 = (dZ2 * W2^T) * derivative_of_activation(z1)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._sigmoid_derivative(self.a1)  # shape: (n_samples, hidden_size)
        
        # Gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / n_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

        # 3. Update parameters
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def compute_loss(self, y_pred, y_one_hot):
        # Categorical Cross-Entropy Loss
        eps = 1e-15
        loss = -np.sum(y_one_hot * np.log(y_pred + eps)) / y_pred.shape[0]
        return loss

    def fit(self, X, y, epochs=1000):
        # Convert integer labels to one-hot encoding
        n_classes = len(np.unique(y))
        y_one_hot = np.eye(n_classes)[y]
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Loss
            loss = self.compute_loss(y_pred, y_one_hot)
            self.loss_history.append(loss)
            
            # Backward pass & update weights
            self.backward(X, y_one_hot)
            
            if epoch % 100 == 0:
                predictions = np.argmax(y_pred, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy*100:.2f}%")


def main():
    # Create simple XOR-like classification problem
    # XOR cannot be solved by a single linear classifier; it needs a hidden layer!
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)  # Labels: 0 and 1

    print("--- Training NumPy MLP on XOR Problem ---")
    # 2 inputs, 4 hidden units, 2 output classes
    nn = ScratchNeuralNetwork(input_size=2, hidden_size=4, output_size=2, learning_rate=0.5)
    nn.fit(X, y, epochs=2000)

    # Validate output
    predictions = nn.forward(X)
    pred_classes = np.argmax(predictions, axis=1)
    print("\n--- Final Predictions ---")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Target: {y[i]} -> Predicted Probabilities: {predictions[i]} -> Predicted Class: {pred_classes[i]}")

if __name__ == '__main__':
    main()
