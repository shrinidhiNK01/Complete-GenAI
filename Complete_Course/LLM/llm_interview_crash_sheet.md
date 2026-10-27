# LLM Interview Crash Sheet

Use this for fast revision.

---

## 1. The 30-Second LLM Story

An LLM is a large transformer-based neural network trained mostly to predict the next token. Text is split into tokens, tokens become embeddings, embeddings pass through transformer blocks with attention and feed-forward layers, and the model outputs probabilities for the next token. Training uses loss, backpropagation, and gradient descent. In real applications, LLMs are often combined with prompts, retrieval, tools, guardrails, evaluation, and monitoring.

---

## 2. Neural Network Basics

Neuron:

- Weighted sum plus bias, then activation.

Formula:

```text
z = w.x + b
a = activation(z)
```

Weights:

- Learned numbers controlling importance.

Bias:

- Shift term.

Layer:

- Group of neurons.

Deep network:

- Many layers.

---

## 3. How Neural Networks Learn

Learning loop:

1. Forward pass.
2. Prediction.
3. Loss calculation.
4. Backpropagation.
5. Gradient descent update.

Loss:

- Measures error.

Gradient:

- Direction of loss change.

Gradient descent:

```text
w = w - learning_rate * gradient
```

Backpropagation:

- Computes gradients using chain rule.

Optimizer:

- Updates weights using gradients.

---

## 4. Backpropagation In One Answer

Backpropagation is an efficient way to calculate how much each weight contributed to the final error. It starts from the loss, moves backward through the network, applies the chain rule, computes gradients, and then an optimizer updates the weights.

Chain rule:

```text
dL/dw = dL/dy * dy/dz * dz/dw
```

---

## 5. LLM Basics

Token:

- Piece of text.

Tokenizer:

- Converts text to token IDs.

Embedding:

- Converts token ID to vector.

Context window:

- Maximum tokens the model can use at once.

Parameter:

- Learned number in the model.

Pretraining:

- Learn general language from huge data.

Fine-tuning:

- Adapt model to task/style/domain.

Inference:

- Use trained model to generate output.

---

## 6. Transformer Basics

Transformer block:

- Attention.
- Residual connection.
- Layer norm.
- Feed-forward network.

Attention:

- Lets tokens communicate.

Self-attention:

- Tokens attend to tokens in the same sequence.

Masked attention:

- Blocks future tokens during generation.

Multi-head attention:

- Multiple attention heads learn different relationships.

Positional encoding:

- Adds order information.

---

## 7. Attention Formula

Q:

- Query, what I am looking for.

K:

- Key, what I contain.

V:

- Value, information to pass.

Formula:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Why divide by sqrt(d_k)?

- To keep scores stable and avoid overly sharp softmax.

---

## 8. Model Types

Encoder-only:

- Good for understanding.
- Example tasks: classification, embeddings, NER.

Decoder-only:

- Good for generation.
- GPT-style.

Encoder-decoder:

- Good for sequence-to-sequence tasks.
- Example: translation, summarization.

---

## 9. How LLMs Store Facts

LLMs store facts in distributed patterns across parameters.

They do not usually store facts like rows in a database.

Attention helps use context.

MLP/feed-forward layers help transform and store learned associations.

Important:

- This is not perfect memory.
- LLMs can hallucinate.

---

## 10. Decoding

Greedy:

- Pick highest probability token.

Temperature:

- Controls randomness.

Top-k:

- Sample from top k tokens.

Top-p:

- Sample from smallest set of tokens whose total probability reaches p.

Beam search:

- Keeps several candidate sequences.

---

## 11. RAG

RAG means Retrieval Augmented Generation.

Steps:

1. Split documents into chunks.
2. Embed chunks.
3. Store in vector database.
4. Embed user query.
5. Retrieve similar chunks.
6. Put context into prompt.
7. Model answers.

Use RAG for:

- Private documents.
- Recent facts.
- Citations.

Use fine-tuning for:

- Style.
- Format.
- Task behavior.

---

## 12. Fine-Tuning

Full fine-tuning:

- Updates all weights.

LoRA:

- Freezes base model and trains small adapter matrices.

QLoRA:

- Quantized base model plus LoRA.

Instruction tuning:

- Teaches model to follow instructions.

Preference tuning:

- Teaches model preferred answers.

---

## 13. Evaluation

Evaluate:

- Correctness.
- Helpfulness.
- Factuality.
- Safety.
- Instruction following.
- Latency.
- Cost.

Metrics:

- Perplexity.
- Accuracy/F1 for closed tasks.
- Human evaluation.
- LLM-as-judge with caution.

For RAG:

- Retrieval quality.
- Faithfulness.
- Citation quality.
- Answer correctness.

---

## 14. AI Images And Videos

Diffusion model:

- Learns to remove noise.

Training:

- Add noise to real image and train model to predict noise.

Generation:

- Start from random noise and denoise step by step.

Latent diffusion:

- Denoises in compressed latent space.

Text-to-image:

- Text encoder guides denoising.

Video generation:

- Denoising across frames with temporal consistency.

Why video is hard:

- Motion, identity consistency, physics, flicker.

---

## 15. Top Interview Q&A

### What is an LLM?

A large language model is a large neural network, usually transformer-based, trained to predict the next token from huge text/code data.

### What is a transformer?

A transformer is a neural architecture using self-attention, feed-forward layers, residual connections, and layer normalization.

### What is attention?

Attention lets each token focus on relevant tokens by comparing query and key vectors, then combining value vectors using softmax weights.

### Why are transformers better than RNNs for LLMs?

Transformers process tokens in parallel during training and handle long-range dependencies better through attention.

### What is tokenization?

Tokenization splits text into smaller units and maps them to IDs.

### What is an embedding?

An embedding is a dense vector representation of a token.

### What is pretraining?

Training a model on large general data, often with next-token prediction.

### What is fine-tuning?

Further training a pretrained model on task-specific or instruction-specific data.

### What is hallucination?

When a model generates plausible but false information.

### How reduce hallucination?

Use RAG, tools, citations, better prompts, evaluation, and refusal when uncertain.

### What is KV cache?

Stored keys and values from previous tokens used to speed up autoregressive generation.

### What is context window?

The maximum number of tokens the model can consider.

### What is LoRA?

A parameter-efficient fine-tuning method that trains small low-rank adapter matrices while freezing the base model.

### What is quantization?

Reducing numerical precision to make models smaller and faster.

### What is diffusion?

A generative method that learns to denoise random noise into images or videos.

---

## 16. Must-Know Code Ideas

Tiny neural network:

```python
z = X @ W + b
a = activation(z)
loss = loss_fn(a, y)
loss.backward()
optimizer.step()
```

Attention:

```python
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

Generation:

```python
for step in range(max_tokens):
    logits = model(context)
    next_token = sample(logits[-1])
    context.append(next_token)
```

RAG:

```python
query -> embedding -> retrieve docs -> prompt with context -> model answer
```

---

## 17. Final Confidence Script

"LLMs are transformer-based next-token predictors. Text becomes tokens, tokens become embeddings, attention lets tokens exchange information, feed-forward layers transform representations, and the output head predicts the next token. Training uses loss, backpropagation, and gradient descent. Real systems add retrieval, tools, evaluation, safety, and monitoring."

