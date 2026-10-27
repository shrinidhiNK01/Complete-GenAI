# Deep Learning Interview Crash Sheet

You said the interview is tomorrow, so this sheet is built for fast survival plus confidence. Read the complete notes first if you have time. If time is short, read this sheet twice and practice answering aloud.

---

## 0. The 30-Second Deep Learning Story

Deep learning uses neural networks with many layers to learn patterns from data. A network takes input, passes it through layers, makes a prediction, compares it with the correct answer using a loss function, then updates weights using backpropagation and an optimizer. It is especially useful for images, text, speech, video, and large datasets because it learns features automatically.

---

## 1. Tonight's Study Order

If you have 6 hours:

1. Neuron, perceptron, MLP, forward propagation.
2. Loss functions, activation functions.
3. Backpropagation, gradient descent, optimizers.
4. Overfitting, dropout, regularization, batch norm, early stopping.
5. CNN: convolution, padding, stride, pooling, transfer learning.
6. RNN, LSTM, GRU.
7. Attention, self-attention, transformers.
8. Code templates.
9. Practice Q&A aloud.

If you have 2 hours:

1. Read sections 0 to 8 of this sheet.
2. Memorize output layer/loss table.
3. Practice the top 25 questions.

If you have 30 minutes:

1. Read the 30-second story.
2. Read the output/loss table.
3. Read CNN, LSTM, transformer summaries.
4. Practice "I know the intuition" style answers.

---

## 2. Core Words You Must Know

Neuron:

- Basic unit that calculates weighted sum plus bias, then activation.

Weight:

- Learns importance of input.

Bias:

- Shifts the activation/decision boundary.

Activation:

- Adds nonlinearity.

Loss:

- Measures prediction error.

Gradient:

- Direction and amount by which loss changes.

Backpropagation:

- Calculates gradients using chain rule.

Optimizer:

- Updates weights using gradients.

Epoch:

- One full pass over training data.

Batch:

- Small group of samples processed together.

Iteration:

- One weight update.

---

## 3. Output Layer and Loss Table

Binary classification:

- Output: 1 sigmoid neuron.
- Loss: binary cross entropy.

Multi-class classification with integer labels:

- Output: softmax with number of classes.
- Loss: sparse categorical cross entropy.

Multi-class classification with one-hot labels:

- Output: softmax with number of classes.
- Loss: categorical cross entropy.

Multi-label classification:

- Output: sigmoid for each label.
- Loss: binary cross entropy.

Regression:

- Output: linear neuron, no activation.
- Loss: MSE, MAE, or Huber.

---

## 4. Activation Functions in One Line

Sigmoid:

- Output 0 to 1; used for binary output; can vanish gradients.

Tanh:

- Output -1 to 1; zero-centered; can vanish gradients.

ReLU:

- Default hidden activation; fast; can have dying ReLU.

Leaky ReLU:

- Allows small negative output; helps dying ReLU.

Softmax:

- Converts class scores into probabilities; used for multi-class output.

---

## 5. Optimizers in One Line

SGD:

- Basic gradient update.

Momentum:

- Smooths updates and speeds movement in consistent directions.

Nesterov:

- Momentum with look-ahead correction.

AdaGrad:

- Adaptive learning rate, but can shrink too much.

RMSProp:

- Adaptive method that fixes AdaGrad's shrinking problem.

Adam:

- Combines momentum and RMSProp; common default.

AdamW:

- Adam with better decoupled weight decay; common in transformers.

---

## 6. Overfitting Diagnosis

Training loss low, validation loss high:

- Overfitting.

Fix:

- More data, augmentation, dropout, L2, early stopping, smaller model.

Training loss high, validation loss high:

- Underfitting.

Fix:

- Bigger model, train longer, better features, reduce regularization.

Loss NaN:

- Learning rate too high, exploding gradients, bad input values.

Fix:

- Lower learning rate, gradient clipping, check NaN/inf, normalize data.

---

## 7. CNN Summary

CNN is used for images because it preserves spatial structure and shares weights.

Convolution:

- Small filter slides over image and detects patterns.

Padding:

- Adds border pixels to control output size.

Stride:

- Step size of filter movement.

Pooling:

- Reduces spatial size and keeps important features.

Transfer learning:

- Use pretrained model like ResNet/MobileNet, replace final layer, train on your dataset.

Interview sentence:

CNNs learn hierarchical visual features: early layers learn edges, middle layers learn textures/shapes, deeper layers learn object parts.

---

## 8. RNN, LSTM, GRU Summary

RNN:

- Sequence model with hidden state memory.

Problem:

- Vanishing/exploding gradients for long sequences.

LSTM:

- Uses cell state and gates to remember long-term information.

LSTM gates:

- Forget gate, input gate, candidate memory, output gate.

GRU:

- Simpler than LSTM; uses update and reset gates.

Bidirectional RNN:

- Reads sequence forward and backward; useful when future context is available.

---

## 9. Attention and Transformer Summary

Attention:

- Lets model focus on important parts of input.

Query:

- What I am looking for.

Key:

- What each item offers.

Value:

- Actual information retrieved.

Self-attention:

- Tokens attend to other tokens in the same sequence.

Multi-head attention:

- Multiple attention heads learn different relationships.

Transformer:

- Uses self-attention, feed-forward layers, residual connections, and layer norm.

Encoder:

- Good for understanding input.

Decoder:

- Good for generation.

Masked attention:

- Prevents looking at future tokens.

---

## 10. Top Interview Q&A

### 1. What is deep learning?

Deep learning is machine learning using neural networks with multiple layers. It learns features automatically and works well for images, text, audio, and large datasets.

### 2. What is a neural network?

A neural network is a collection of connected layers. Each layer transforms input data into more useful representations for prediction.

### 3. What is a perceptron?

A perceptron is the simplest binary classifier. It calculates a weighted sum and applies a step function to decide class 0 or 1.

### 4. Why can a perceptron not solve XOR?

Because XOR is not linearly separable. A single perceptron can only create one linear decision boundary.

### 5. What is an MLP?

An MLP is a feed-forward neural network with input, hidden, and output layers. Hidden layers allow it to learn nonlinear patterns.

### 6. Why do we need activation functions?

Without activation functions, multiple layers would still behave like one linear layer. Activations allow neural networks to learn complex nonlinear functions.

### 7. What is forward propagation?

Forward propagation is the process of passing input through the network to generate prediction and calculate loss.

### 8. What is backpropagation?

Backpropagation calculates gradients of loss with respect to weights using the chain rule.

### 9. Is backpropagation the same as gradient descent?

No. Backpropagation computes gradients. Gradient descent or another optimizer updates weights using those gradients.

### 10. What is learning rate?

Learning rate controls how large each weight update is. Too high can diverge; too low trains slowly.

### 11. What is vanishing gradient?

It happens when gradients become very small in deep networks, so early layers learn slowly.

### 12. How do you fix vanishing gradients?

Use ReLU, proper initialization, normalization, residual connections, LSTM/GRU for sequences.

### 13. What is exploding gradient?

It happens when gradients become too large, causing unstable training or NaN loss.

### 14. How do you fix exploding gradients?

Use gradient clipping, lower learning rate, better initialization, and normalization.

### 15. Why not initialize all weights to zero?

All neurons would learn the same features because they receive identical gradients. Random initialization breaks symmetry.

### 16. What is Xavier initialization?

It initializes weights so variance remains stable across layers, usually good for sigmoid/tanh.

### 17. What is He initialization?

It is initialization designed for ReLU-based networks.

### 18. What is dropout?

Dropout randomly disables neurons during training to reduce overfitting.

### 19. Is dropout active during inference?

No. Dropout is active during training and disabled during inference.

### 20. What is batch normalization?

Batch normalization normalizes activations using mini-batch statistics and learns scale and shift.

### 21. Why use batch normalization?

It stabilizes and speeds training, allows higher learning rates, and adds mild regularization.

### 22. What is regularization?

Regularization adds constraints or penalties to reduce overfitting.

### 23. Difference between L1 and L2?

L1 can make weights zero and encourages sparsity. L2 keeps weights small and is commonly used in neural networks.

### 24. What is early stopping?

Stopping training when validation loss stops improving.

### 25. What is CNN?

CNN is a neural network designed for spatial data like images. It uses convolution filters to learn local patterns.

### 26. What is convolution?

A filter slides over input and computes dot products to detect patterns like edges or textures.

### 27. What is padding?

Padding adds pixels around the image to control output size and preserve border information.

### 28. What is stride?

Stride is how far the filter moves each step.

### 29. What is pooling?

Pooling reduces feature map size. Max pooling keeps the strongest activation in a region.

### 30. Why CNN instead of ANN for images?

ANN has too many parameters and ignores spatial structure. CNN uses local connectivity and weight sharing.

### 31. What is transfer learning?

Using a pretrained model and adapting it to a new task.

### 32. Feature extraction vs fine tuning?

Feature extraction freezes pretrained base and trains a new head. Fine tuning unfreezes some layers and trains them with a small learning rate.

### 33. What is RNN?

RNN is a neural network for sequence data. It maintains hidden state memory.

### 34. What is BPTT?

Backpropagation Through Time trains RNNs by unrolling them over sequence steps and applying backpropagation.

### 35. Why LSTM?

LSTM handles long-term dependencies better than simple RNN using gates and cell state.

### 36. LSTM vs GRU?

LSTM is more complex with separate cell state. GRU is simpler and faster, often with similar performance.

### 37. What is attention?

Attention computes weights over input parts so the model can focus on relevant information.

### 38. What are Q, K, V?

Query is what we search with, key is what we compare against, value is the information returned.

### 39. Why scale attention by sqrt(d_k)?

To prevent large dot products from making softmax too sharp and gradients too small.

### 40. What is multi-head attention?

It runs multiple attention operations in parallel so the model can learn different relationships.

### 41. What is positional encoding?

It adds order information because attention alone does not know token position.

### 42. What is transformer?

A transformer is a neural architecture based on self-attention, feed-forward layers, residual connections, and layer normalization.

### 43. Encoder vs decoder transformer?

Encoder understands input. Decoder generates output. Encoder-decoder models transform one sequence into another.

### 44. What is masked self-attention?

It blocks future tokens so generation only depends on past tokens.

### 45. What is cross-attention?

Decoder queries attend to encoder outputs to use source sequence information.

### 46. What is data augmentation?

Creating modified training examples to improve generalization, like image flips and rotations.

### 47. What is data leakage?

When information from validation/test data enters training. It gives falsely high performance.

### 48. Why split train, validation, and test?

Train learns weights, validation tunes choices, test gives final unbiased performance.

### 49. Accuracy vs precision vs recall?

Accuracy is overall correctness. Precision measures correctness among predicted positives. Recall measures how many actual positives were found.

### 50. What is F1 score?

F1 is the harmonic mean of precision and recall, useful for imbalanced data.

---

## 11. Code Snippets to Memorize

### Binary ANN

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

### CNN

```python
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(num_classes, activation="softmax")
])
```

### LSTM

```python
model = keras.Sequential([
    layers.Embedding(vocab_size, 128),
    layers.LSTM(64),
    layers.Dense(1, activation="sigmoid")
])
```

### Early Stopping

```python
callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

---

## 12. How to Answer When You Do Not Know

Use this pattern:

"I do not remember the exact formula, but I understand the intuition. The purpose of this concept is ____. It works by ____. In practice, we use it when ____."

Example:

"I do not remember the exact LSTM equations, but I understand the intuition. LSTM solves the long-term dependency problem in RNNs. It uses gates to decide what to forget, what to store, and what to output."

That answer is much better than going silent.

---

## 13. Final Confidence Script

Before the interview, say this aloud:

"Deep learning is not magic. It is layers, activations, loss, gradients, and optimization. CNNs are for spatial patterns. RNN/LSTM/GRU are for sequences. Attention and transformers are for learning relationships across tokens efficiently. If I can explain what problem each concept solves, I can handle the interview."

