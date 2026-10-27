# Deep Learning: The Complete Masterclass (Beginner to Advanced)

Welcome to the ultimate guide to Deep Learning (DL). This document explains every major deep learning concept, architecture, and mathematical intuition in simple English, followed by PyTorch step-by-step guides.

---

## Table of Contents
1. **Introduction to Deep Learning**
2. **Anatomy of a Neural Network**
3. **Forward Propagation & Activation Functions**
4. **Loss Functions**
5. **The Math of Backpropagation (Chain Rule)**
6. **Optimizers: SGD, Momentum, RMSprop, and Adam**
7. **Convolutional Neural Networks (CNNs)**
8. **Recurrent Neural Networks (RNNs) & LSTMs**
9. **Transformers & The Attention Mechanism**
10. **Regularization & Normalization (Dropout, Batch Norm, Layer Norm)**
11. **Introduction to PyTorch Workflow**

---

## 1. Introduction to Deep Learning

### What is Deep Learning?
Deep Learning is a subfield of Machine Learning inspired by the structure and function of the human brain. While traditional ML models require humans to manually design features (e.g., extract edges or textures from an image), Deep Learning models learn **feature hierarchies** automatically directly from raw data (e.g., raw pixels).

*   **Shallow Learning:** Input $\to$ Manual Feature Engineering $\to$ Linear Model $\to$ Prediction
*   **Deep Learning:** Input $\to$ Many layers of Neurons (Feature learning + Classification) $\to$ Prediction

---

## 2. Anatomy of a Neural Network

A neural network is composed of layers containing interconnected nodes called **Neurons**.

```text
Input Layer          Hidden Layer          Output Layer
   (Features)         (Intermediary)         (Predictions)
     [X1] -------------> [H1] -------------> [Y_hat]
            \          /      \            /
             \        /        \          /
     [X2] -------------> [H2] -----------/
```

*   **Input Layer:** Takes raw features (e.g., pixel intensities, tabular numbers).
*   **Hidden Layers:** Intermediate layers that extract patterns. "Deep" networks have many hidden layers.
*   **Output Layer:** Returns the final prediction (e.g., classification probabilities, continuous value).
*   **Weights ($W$):** Connection strengths between neurons. They determine the importance of input signals.
*   **Biases ($b$):** Offset values added to inputs. They allow the activation function to shift left or right.

---

## 3. Forward Propagation & Activation Functions

### Forward Propagation
For each neuron, we calculate a weighted sum of inputs and add a bias, then pass it through an activation function.

$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = W \cdot X + b$$
$$a = \text{Activation}(z)$$

### Why do we need Activation Functions?
Without activation functions, every layer is just a linear equation. Multiplying multiple linear equations together just results in another linear equation ($y = mx + c$). Activation functions introduce **non-linearity**, allowing the network to learn complex, curvy, non-linear patterns (like faces, text, and audio).

### Common Activation Functions

| Activation Function | Math Formula | Properties & When to Use |
| :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Outputs $[0, 1]$. Used in output layers for binary classification. Vulnerable to vanishing gradients. |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Outputs $[-1, 1]$. Zero-centered, which makes optimization easier. Vulnerable to vanishing gradients. |
| **ReLU** | $f(z) = \max(0, z)$ | Outputs $0$ for negative values, $z$ for positive. Extremely fast, stops vanishing gradients. Standard choice for hidden layers. |
| **Leaky ReLU** | $f(z) = \max(0.01z, z)$ | Solves the "dying ReLU" problem (where negative weights get stuck at zero) by providing a tiny slope for negative inputs. |
| **GELU** | $f(z) = z \cdot \Phi(z)$ | Gaussian Error Linear Unit. Scales inputs by their cumulative distribution probability. Standard in modern Transformers (BERT, GPT). |
| **Softmax** | $f(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}$ | Squashes a vector of scores into a probability distribution summing to 1. Used in multi-class output layers. |

---

## 4. Loss Functions

The loss function measures the discrepancy between predictions ($\hat{y}$) and true labels ($y$).

*   **Mean Squared Error (MSE):** For Regression.
    $$L = \frac{1}{N} \sum (\hat{y}_i - y_i)^2$$
*   **Binary Cross-Entropy (BCE):** For Binary Classification (spam vs. not spam).
    $$L = -\frac{1}{N} \sum [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$
*   **Categorical Cross-Entropy:** For Multi-Class Classification (cat vs. dog vs. bird).
    $$L = -\sum y_i \log(\hat{y}_i)$$

---

## 5. The Math of Backpropagation (Chain Rule)

Backpropagation is how a neural network learns. It calculates how much each weight and bias contributed to the final loss.

### The Chain Rule
If variable $y$ depends on $u$, and $u$ depends on $x$, then the rate of change of $y$ with respect to $x$ is:

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

### Step-by-Step Backpropagation
Let's consider a single neuron network:
$$\text{Input } X \to [z = W \cdot X + b] \to [a = \sigma(z)] \to \text{Loss } L$$

We want to find how much $W$ affects $L$, which is $\frac{\partial L}{\partial W}$.
By the Chain Rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W}$$

1.  $\frac{\partial L}{\partial a}$: How loss changes with prediction change.
2.  $\frac{\partial a}{\partial z}$: The derivative of the activation function w.r.t the weighted sum.
3.  $\frac{\partial z}{\partial W}$: The derivative of the linear summation w.r.t weights (which equals $X$).

Multiplying these gives the exact gradient. We then update the weight:
$$W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$$

*See complete NumPy from-scratch implementation in: `implementations/neural_network_from_scratch.py`*

---

## 6. Optimizers: Moving Down the Slope

Optimizers update weights based on calculated gradients.

*   **Stochastic Gradient Descent (SGD):** Updates weights using a single sample or small batch. Can be noisy and get stuck in local minima.
*   **Momentum:** Adds a fraction of the previous step's update vector to the current update. Like a heavy ball rolling down a hill, it gains momentum and rolls past small bumps (local minima).
*   **RMSprop:** Scales the learning rate for each parameter based on its historical gradients, preventing weights with huge gradients from exploding.
*   **Adam (Adaptive Moment Estimation):** Combines Momentum and RMSprop. It tracks both the average of past gradients (first moment) and the average of past *squared* gradients (second moment). It is the default optimizer for most deep learning models.

---

## 7. Convolutional Neural Networks (CNNs)

CNNs are designed specifically to process grid-structured data like images.

```text
Raw Image  ===> [ Convolution Layers ] ===> [ Pooling Layers ] ===> [ Fully Connected ] ===> Class
                  (Detects edges/shapes)       (Shrinks size)          (Decision making)
```

### Core Operations
1.  **Convolution:** A small matrix called a **Filter** or **Kernel** slides across the image, multiplying its values with local pixel values. This extracts feature maps (like horizontal lines, curves, or textures).
2.  **Stride:** The number of pixels the filter shifts at each step. Stride = 1 means shift by 1 pixel, Stride = 2 halves the output dimensions.
3.  **Padding:** Adding dummy pixels (usually zeros) around the edges of the image so that the filter doesn't shrink the image size too quickly.
4.  **Pooling:** Downsampling operation. **Max Pooling** takes the maximum value in a window (e.g., $2\times2$), preserving the most prominent features while reducing the computational load.

*See complete CNN implementation in: `implementations/cnn_mnist.py`*

---

## 8. Recurrent Neural Networks (RNNs) & LSTMs

### RNNs
Traditional networks assume all inputs are independent. RNNs are designed for sequential data (like time-series, text, or audio). They pass information from one step to the next using a **Hidden State** ($h_t$), acting as a memory.

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

#### The Limitation: Vanishing Gradients
When backpropagating through long sequences, gradients are multiplied repeatedly. If the weights are small, the gradient shrinks to zero (vanishes), meaning the model forgets long-term dependencies.

---

### LSTMs (Long Short-Term Memory)
LSTMs solve vanishing gradients using a **Cell State** ($C_t$) that runs straight down the chain with only linear interactions, and **Gates** that regulate what information is added or removed.

```text
Forget Gate (Remove old facts) -> Input Gate (Add new facts) -> Output Gate (Decide next hidden state)
```

1.  **Forget Gate:** Decides what to throw away from the past cell state.
    $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2.  **Input Gate:** Decides which new information to store in the cell state.
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
3.  **Output Gate:** Decides what the next hidden state ($h_t$) should be.
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

---

## 9. Transformers & The Attention Mechanism

Introduced in the paper *"Attention Is All You Need"* (2017), the Transformer replaced RNNs for NLP by processing all tokens in a sequence simultaneously instead of one by one.

### Self-Attention
Self-attention allows the model to look at other words in the input sequence to help encode a specific word better. E.g., in *"The animal didn't cross the street because it was too tired"*, the word **"it"** refers to the **"animal"**, not the **"street"**. Self-attention calculates this association.

#### The Formula
For each word, we project its embedding into three vectors:
*   **Query ($Q$):** What we are looking for.
*   **Key ($K$):** The labels/identifiers of all other words.
*   **Value ($V$):** The actual content of the words.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

1.  **Dot Product ($QK^T$):** Scores how much attention each word should pay to every other word.
2.  **Scale ($\sqrt{d_k}$):** Divides by the square root of the Key vector dimension to prevent gradients from blowing up.
3.  **Softmax:** Turns scores into probabilities summing to $1$.
4.  **Multiply by $V$:** Multiplies probabilities by the values to get the weighted representations.

### Multi-Head Attention
Instead of performing self-attention once, we perform it multiple times in parallel with different weight matrices (heads). Each head can focus on different aspects of the sequence (e.g., one head for grammar, one for pronouns, one for actions).

### Positional Encoding
Since Transformers process all words at once, they have no built-in notion of word order. We add a **Positional Encoding** (sine and cosine wave values) to the word embeddings to give the model spatial context.

*See complete Self-Attention code in: `implementations/transformer_attention.py`*

---

## 10. Regularization & Normalization

### Dropout
During training, randomly "turn off" (zero out) a percentage of neurons (e.g., 20%). This forces the remaining neurons to learn redundant features and prevents them from co-adapting too closely, reducing overfitting.

### Batch Normalization (Batch Norm)
Normalizes the inputs to each layer *across the batch dimension*.
*   **Benefits:** Speeds up training, allows higher learning rates, acts as a mild regularizer.
*   **Where:** Usually placed before activation functions in feedforward/CNN networks.

### Layer Normalization (Layer Norm)
Normalizes the inputs to each layer *across the feature dimension* for each individual sample.
*   **Benefits:** Doesn't depend on batch size. Perfect for sequential models (RNNs, Transformers).

---

## 11. Introduction to PyTorch Workflow

To build a model in PyTorch:
1.  **Define Dataset:** Inherit from `torch.utils.data.Dataset` and implement `__len__` and `__getitem__`.
2.  **Wrap in DataLoader:** Set batch sizes and shuffle settings.
3.  **Create Model Architecture:** Inherit from `torch.nn.Module`, define layers in `__init__`, and write routing logic in `forward`.
4.  **Define Loss and Optimizer:** Select an optimizer (e.g., `torch.optim.Adam`) and loss function (e.g., `torch.nn.CrossEntropyLoss`).
5.  **Write Training Loop:**
    *   Call `optimizer.zero_grad()` to clear previous gradients.
    *   Pass inputs through model: `outputs = model(inputs)`.
    *   Calculate loss: `loss = criterion(outputs, labels)`.
    *   Perform backpropagation: `loss.backward()`.
    *   Update weights: `optimizer.step()`.
