# Complete LLM Documentation: Beginner to Advanced

This document explains large language models from the ground up in simple English. It follows the path in your video list:

- What is a neural network?
- How gradient descent helps neural networks learn.
- Backpropagation intuition.
- Backpropagation calculus.
- What are LLMs?
- Transformers.
- Attention step by step.
- How LLMs may store facts.
- How AI images and videos work.

It also goes beyond the list and covers tokenization, embeddings, transformer internals, training, fine-tuning, RAG, inference, decoding, evaluation, deployment, quantization, safety, multimodal models, and code patterns.

---

## 1. The Big Picture

An LLM is a large neural network trained mostly to predict the next token.

Example:

```text
Input: The capital of France is
Target next token: Paris
```

After seeing billions or trillions of tokens, the model learns patterns in language, facts, reasoning styles, code structure, grammar, and world knowledge.

Important idea:

An LLM does not store a database table like:

```text
France -> Paris
```

Instead, it stores knowledge inside billions of learned numbers called parameters. These parameters shape how the model transforms input tokens into probability distributions over next tokens.

---

## 2. What Is A Neural Network?

A neural network is a mathematical model made of layers.

Each layer takes numbers in, transforms them, and passes numbers forward.

The basic unit is a neuron.

### 2.1 A Single Neuron

A neuron receives inputs:

```text
x1, x2, x3
```

Each input has a weight:

```text
w1, w2, w3
```

The neuron calculates:

```text
z = w1*x1 + w2*x2 + w3*x3 + b
```

Then it applies an activation function:

```text
a = activation(z)
```

Where:

- `x` means input.
- `w` means weight.
- `b` means bias.
- `z` means raw score before activation.
- `a` means activated output.

### 2.2 What Are Weights?

Weights are learned numbers.

They decide how strongly one input affects the output.

Large positive weight:

- Input increases output strongly.

Large negative weight:

- Input decreases output strongly.

Weight near zero:

- Input has little effect.

### 2.3 What Is Bias?

Bias shifts the neuron's output.

It is similar to the intercept in a line equation:

```text
y = mx + c
```

Without bias, the model has less flexibility.

### 2.4 Why Many Neurons?

One neuron can learn only a simple pattern.

Many neurons together can learn complicated patterns.

Example:

- One neuron may detect one simple feature.
- A layer may detect many features.
- Many layers can combine simple features into complex concepts.

### 2.5 Why Many Layers?

Layers allow hierarchical learning.

For images:

- Early layers detect edges.
- Middle layers detect shapes.
- Later layers detect objects.

For text:

- Early layers learn token relationships.
- Middle layers learn phrase and syntax patterns.
- Later layers learn task and semantic patterns.

---

## 3. How Neural Networks Learn

Neural networks learn by changing weights to reduce error.

The learning loop:

1. Put input into the model.
2. Model makes prediction.
3. Compare prediction with correct answer.
4. Calculate loss.
5. Find how each weight affected the loss.
6. Change weights slightly to reduce loss.
7. Repeat many times.

This process is training.

### 3.1 Prediction

If the model sees:

```text
The sky is
```

It may output probabilities:

```text
blue: 0.60
clear: 0.12
falling: 0.01
```

The model's prediction is a probability distribution over possible next tokens.

### 3.2 Loss

Loss measures how wrong the model is.

If the correct next token is `blue` and model gives high probability to `blue`, loss is low.

If the model gives low probability to `blue`, loss is high.

For language models, the common loss is cross entropy.

### 3.3 Optimization

Optimization means improving model parameters to reduce loss.

The most important optimization idea is gradient descent.

---

## 4. Gradient Descent

Gradient descent is the basic idea behind how neural networks learn.

Imagine you are standing on a mountain in fog and want to reach the lowest valley. You cannot see far, but you can feel which direction slopes downward. You take a small step downhill, then repeat.

In machine learning:

- Mountain height = loss.
- Location = model weights.
- Downhill direction = negative gradient.
- Step size = learning rate.

### 4.1 Gradient

A gradient tells how loss changes when a parameter changes.

If a weight has positive gradient:

- Increasing that weight increases loss.
- So we decrease the weight.

If a weight has negative gradient:

- Increasing that weight decreases loss.
- So we increase the weight.

### 4.2 Update Rule

```text
new_weight = old_weight - learning_rate * gradient
```

### 4.3 Learning Rate

Learning rate controls the size of each step.

Too high:

- Model may jump over the minimum.
- Loss may explode or become unstable.

Too low:

- Training is very slow.

### 4.4 Batch, Mini-Batch, Epoch

Training data is usually processed in mini-batches.

Batch:

- A group of examples processed together.

Iteration:

- One parameter update.

Epoch:

- One full pass through the training data.

Example:

```text
Training examples = 1000
Batch size = 100
Iterations per epoch = 10
```

---

## 5. Backpropagation Intuition

Backpropagation answers this question:

```text
Which weights caused the error, and by how much?
```

The model first moves forward:

```text
input -> prediction -> loss
```

Then the error signal moves backward:

```text
loss -> output layer -> hidden layers -> input-side layers
```

Each layer receives information about how it contributed to the final error.

### 5.1 Simple Intuition

Imagine a factory line:

```text
Worker A -> Worker B -> Worker C -> Final product
```

If the final product is bad, we need to know:

- Did Worker C make the main mistake?
- Did Worker B send bad intermediate work?
- Did Worker A start the error?

Backpropagation distributes blame backward through the network.

### 5.2 Backpropagation vs Optimizer

Backpropagation:

- Computes gradients.

Optimizer:

- Uses gradients to update weights.

Do not confuse them.

---

## 6. Backpropagation Calculus

Backpropagation uses the chain rule.

### 6.1 Chain Rule

If:

```text
L depends on y
y depends on z
z depends on w
```

Then:

```text
dL/dw = dL/dy * dy/dz * dz/dw
```

This tells us how changing `w` changes the final loss `L`.

### 6.2 Simple One-Neuron Example

Forward pass:

```text
z = w*x + b
y_hat = z
L = (y_hat - y)^2
```

Find gradients:

```text
dL/dy_hat = 2 * (y_hat - y)
dy_hat/dz = 1
dz/dw = x
dz/db = 1
```

So:

```text
dL/dw = 2 * (y_hat - y) * x
dL/db = 2 * (y_hat - y)
```

Update:

```text
w = w - learning_rate * dL/dw
b = b - learning_rate * dL/db
```

### 6.3 Why Deep Networks Need Backpropagation

A deep network has many layers:

```text
x -> layer1 -> layer2 -> layer3 -> output -> loss
```

The first layer affects the loss indirectly through every later layer.

Backpropagation efficiently applies the chain rule from the end to the beginning.

### 6.4 Computational Graph

A computational graph breaks a calculation into small operations.

Example:

```text
x, w -> multiply -> z
z, b -> add -> a
a, y -> loss -> L
```

During backpropagation, each operation knows how to send gradients to its inputs.

This is how frameworks like PyTorch and TensorFlow calculate gradients automatically.

---

## 7. From Neural Networks To Language Models

A language model predicts text.

The simplest language model predicts the next word/token from previous words/tokens.

Example:

```text
Input tokens: I love machine
Next token: learning
```

### 7.1 What Is A Token?

A token is a piece of text.

It may be:

- A word.
- Part of a word.
- A character.
- Punctuation.
- Space-like marker.

Example:

```text
"unbelievable" -> "un", "believ", "able"
```

Modern LLMs usually use subword tokens.

### 7.2 Why Not Use Words Directly?

Word-level tokenization has problems:

- Huge vocabulary.
- Unknown words.
- Hard to handle misspellings and rare words.

Character-level tokenization:

- Small vocabulary.
- But sequences become very long.

Subword tokenization is a practical balance.

### 7.3 Common Tokenization Methods

Byte Pair Encoding:

- Starts with characters or bytes.
- Repeatedly merges common pairs.

WordPiece:

- Similar subword method used in BERT-style models.

SentencePiece:

- Treats text as raw stream and learns subword units.

Byte-level tokenization:

- Can represent almost any text.

---

## 8. Embeddings

Neural networks need numbers, not raw text.

An embedding converts each token into a vector of numbers.

Example:

```text
token "king" -> [0.21, -0.55, 0.84, ...]
```

### 8.1 Why Embeddings Work

During training, tokens used in similar contexts get similar vectors.

Example:

- `cat` and `dog` may be close.
- `king` and `queen` may be related.
- `Paris` and `France` may develop a relationship.

### 8.2 Embedding Matrix

If vocabulary size is 50,000 and embedding dimension is 768:

```text
embedding matrix shape = 50000 x 768
```

Each row is the vector for one token.

### 8.3 Contextual Embeddings

Static embeddings:

- One vector per word.
- Example: Word2Vec.

Contextual embeddings:

- Token vector changes based on surrounding text.
- Example: transformers.

The word `bank` should mean different things in:

```text
river bank
bank account
```

Transformers create context-aware representations.

---

## 9. Why Transformers?

Before transformers, sequence models often used RNNs and LSTMs.

RNN problem:

- Processes tokens one by one.
- Hard to parallelize.
- Struggles with very long-range dependencies.

Transformer advantage:

- Uses attention.
- Processes all tokens in parallel during training.
- Handles long-range relationships better.
- Scales very well.

---

## 10. Attention Step By Step

Attention lets each token decide which other tokens are important.

Example:

```text
The animal did not cross the road because it was tired.
```

The token `it` should pay attention to `animal`.

### 10.1 Query, Key, Value

Attention uses:

Query:

- What am I looking for?

Key:

- What information do you contain?

Value:

- What information should be passed along?

For every token, the model creates:

```text
Q = query vector
K = key vector
V = value vector
```

These are learned projections of token representations.

### 10.2 Attention Scores

To see how much token A should attend to token B:

```text
score = dot(query_A, key_B)
```

Large score:

- Token B is relevant to token A.

Small score:

- Token B is less relevant.

### 10.3 Scaling

Dot products can get large when vector dimension is large.

Large dot products can make softmax too sharp.

So we scale:

```text
scaled_score = score / sqrt(d_k)
```

### 10.4 Softmax

Softmax converts scores into weights that sum to 1.

Example:

```text
scores  = [2.0, 1.0, 0.0]
weights = [0.66, 0.24, 0.10]
```

### 10.5 Weighted Sum

The final attention output is:

```text
output = weights * values
```

More exactly:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### 10.6 Self-Attention

Self-attention means Q, K, and V all come from the same sequence.

Every token attends to tokens in its own context.

### 10.7 Causal / Masked Self-Attention

For text generation, the model must not see future tokens.

Example:

When predicting token 4, it can only use tokens 1, 2, and 3.

So we apply a mask that blocks future positions.

This is used in GPT-style models.

---

## 11. Multi-Head Attention

One attention head can learn one kind of relationship.

Multiple heads let the model learn many relationships at once.

One head may track:

- Grammar.

Another may track:

- Names and references.

Another may track:

- Long-range topic relationships.

Process:

1. Project input into Q, K, V for each head.
2. Run attention separately in each head.
3. Concatenate head outputs.
4. Apply output projection.

---

## 12. Positional Information

Attention alone does not know order.

The set of words:

```text
dog bites man
man bites dog
```

contains the same words but different meaning.

So transformers add position information.

### 12.1 Sinusoidal Positional Encoding

Original transformer used sine and cosine waves at different frequencies.

Benefit:

- Can represent relative positions.
- Works for different sequence lengths.

### 12.2 Learned Positional Embeddings

The model learns a vector for each position.

Simple and common.

### 12.3 Rotary Position Embeddings

RoPE rotates query/key vectors based on position.

Common in many modern LLMs.

Good for:

- Relative position information.
- Long context behavior.

---

## 13. Transformer Block

A transformer block usually contains:

1. Layer normalization.
2. Multi-head attention.
3. Residual connection.
4. Layer normalization.
5. Feed-forward network.
6. Residual connection.

Simplified:

```text
x -> attention -> add residual -> feed-forward -> add residual
```

### 13.1 Residual Connection

Residual connection means:

```text
output = x + layer(x)
```

Why useful:

- Helps gradients flow.
- Makes deep models easier to train.

### 13.2 Layer Normalization

Layer norm normalizes features within each token representation.

It helps training stability.

Transformers usually use layer norm instead of batch norm.

### 13.3 Feed-Forward Network

After attention, each token goes through an MLP.

Typical shape:

```text
d_model -> 4*d_model -> d_model
```

This adds nonlinear transformation.

---

## 14. Transformer Architectures

### 14.1 Encoder-Only Models

Examples:

- BERT-style models.

Good for:

- Classification.
- Named entity recognition.
- Sentence embeddings.
- Understanding tasks.

Training objective:

- Often masked language modeling.

### 14.2 Decoder-Only Models

Examples:

- GPT-style models.

Good for:

- Text generation.
- Chatbots.
- Code generation.

Training objective:

- Next-token prediction.

Uses:

- Causal masked self-attention.

### 14.3 Encoder-Decoder Models

Examples:

- T5-style models.
- Original translation transformer.

Good for:

- Translation.
- Summarization.
- Input-output sequence tasks.

Architecture:

- Encoder reads input.
- Decoder generates output.
- Decoder uses cross-attention over encoder outputs.

---

## 15. What Is A Large Language Model?

An LLM is a language model with many parameters trained on huge text/code datasets.

Important parts:

- Tokenizer.
- Embedding layer.
- Many transformer blocks.
- Output head that predicts next token.

Training:

```text
Input: tokens 1 to n-1
Target: tokens 2 to n
```

The model learns to predict every next token in the sequence.

### 15.1 Parameters

Parameters are learned numbers:

- Embedding values.
- Attention projection weights.
- Feed-forward network weights.
- Output head weights.

Model size examples:

```text
Millions of parameters: small model
Billions of parameters: large model
Trillions of parameters: extremely large model
```

### 15.2 Why Scale Helps

With more data, parameters, and compute, models often learn better general patterns.

Scaling can improve:

- Grammar.
- Factual recall.
- Reasoning patterns.
- Translation.
- Code.
- Instruction following.

But scaling alone does not solve:

- Hallucination.
- Bias.
- Poor grounding.
- Need for private/latest knowledge.

---

## 16. Training An LLM

LLM training usually has stages.

### 16.1 Pretraining

The model learns general language by predicting tokens from huge datasets.

Data may include:

- Web pages.
- Books.
- Code.
- Articles.
- Documentation.
- Conversations.

Objective:

- Next-token prediction for decoder-only models.

### 16.2 Supervised Fine-Tuning

After pretraining, the model may be trained on examples of instructions and good responses.

Example:

```text
User: Explain gradient descent simply.
Assistant: Gradient descent is...
```

This teaches the model to follow instructions better.

### 16.3 Preference Tuning

The model can be trained using human or AI preferences.

Goal:

- Make responses more helpful, harmless, and aligned with user expectations.

Common approaches:

- RLHF.
- DPO.
- Other preference optimization methods.

### 16.4 Domain Fine-Tuning

Train the model further on domain-specific data.

Examples:

- Medical notes.
- Legal documents.
- Company support tickets.
- Codebase-specific examples.

Risk:

- Overfitting.
- Forgetting general ability.
- Leaking private data if not handled carefully.

---

## 17. How LLMs Store Facts

This is a deep and active research area, so we should speak carefully.

Simple idea:

LLMs store facts in distributed patterns across parameters, not in one obvious memory cell.

Example fact:

```text
Paris is the capital of France.
```

The model sees many text patterns:

```text
France's capital, Paris...
Paris, the capital of France...
The capital city of France is Paris...
```

Training adjusts parameters so that when context points toward:

```text
The capital of France is
```

the output probability for `Paris` becomes high.

### 17.1 Distributed Representation

A fact may be represented across:

- Token embeddings.
- Attention heads.
- MLP neurons.
- Layer interactions.

It is not usually stored as one simple line.

### 17.2 Attention vs MLP Memory

Attention helps retrieve information from the current context.

MLP layers may store learned associations and transformations in parameters.

A simplified view:

- Attention moves information between tokens.
- MLPs process and transform information at each token position.

### 17.3 Why LLMs Hallucinate

LLMs generate likely text, not guaranteed truth.

They may hallucinate when:

- Training data had conflicting information.
- The question asks for unknown or recent facts.
- The model overgeneralizes patterns.
- The prompt pressures it to answer.
- No external source is provided.

Solutions:

- Retrieval augmented generation.
- Tool use.
- Citations.
- Calibration.
- Refusal when uncertain.
- Better evaluation and prompting.

---

## 18. Inference

Inference means using a trained model to generate output.

Process:

1. Tokenize prompt.
2. Run model.
3. Get probability distribution for next token.
4. Choose next token using decoding strategy.
5. Append token to context.
6. Repeat.

### 18.1 Greedy Decoding

Always choose highest probability token.

Pros:

- Deterministic.
- Simple.

Cons:

- Can be repetitive or dull.

### 18.2 Sampling

Sample from probability distribution.

Pros:

- More creative.

Cons:

- Can be less reliable.

### 18.3 Temperature

Temperature controls randomness.

Low temperature:

- More focused.
- More deterministic.

High temperature:

- More random.
- More creative but riskier.

### 18.4 Top-k Sampling

Only sample from top k tokens.

### 18.5 Top-p / Nucleus Sampling

Sample from smallest set of tokens whose probabilities add up to p.

Example:

```text
top_p = 0.9
```

### 18.6 Beam Search

Keeps multiple candidate sequences.

Often useful for translation/summarization.

Less common for open-ended chat.

---

## 19. Context Window

The context window is the maximum number of tokens the model can consider at once.

If context window is 8,000 tokens, the model can only directly use that many input and generated tokens.

Long context problems:

- More memory usage.
- More compute.
- Harder retrieval from long prompt.
- Important details can be ignored.

Techniques:

- Longer positional encoding.
- Sparse attention.
- Retrieval.
- Summarization.
- Chunking.

---

## 20. KV Cache

During generation, the model produces one token at a time.

Without caching, it would recalculate attention keys and values for all previous tokens every step.

KV cache stores previous keys and values.

Benefit:

- Faster generation.

Cost:

- More memory.

---

## 21. Retrieval Augmented Generation

RAG gives the model external documents at answer time.

Process:

1. Split documents into chunks.
2. Convert chunks to embeddings.
3. Store embeddings in vector database.
4. User asks question.
5. Convert question to embedding.
6. Retrieve most similar chunks.
7. Put chunks into prompt.
8. Model answers using retrieved context.

Why RAG is useful:

- Adds private knowledge.
- Adds recent knowledge.
- Reduces hallucination.
- Avoids expensive fine-tuning for factual updates.

RAG risks:

- Bad chunking.
- Poor retrieval.
- Irrelevant context.
- Model ignores context.
- No citation/grounding checks.

Good RAG systems need:

- Clean documents.
- Good chunk size.
- Good embeddings.
- Metadata filters.
- Reranking.
- Evaluation.
- Source citations.

---

## 22. Fine-Tuning vs RAG

Use RAG when:

- Need factual knowledge from documents.
- Knowledge changes often.
- Need citations.
- Want to avoid retraining.

Use fine-tuning when:

- Need style/format behavior.
- Need domain-specific response pattern.
- Need task-specific transformation.
- Have many high-quality examples.

Do not fine-tune just to teach many facts that change often. RAG is usually better for that.

---

## 23. Parameter-Efficient Fine-Tuning

Full fine-tuning updates all model weights.

This is expensive.

Parameter-efficient fine-tuning updates a small number of extra parameters.

### 23.1 LoRA

LoRA stands for Low-Rank Adaptation.

Idea:

- Freeze original model.
- Add small trainable low-rank matrices.
- Train only those matrices.

Benefits:

- Less memory.
- Faster.
- Easier to store multiple adapters.

### 23.2 QLoRA

QLoRA combines quantization with LoRA.

Idea:

- Load base model in low precision.
- Train small LoRA adapters.

Useful for limited GPU memory.

---

## 24. Quantization

Quantization reduces number precision.

Example:

- FP32 uses 32-bit floating point.
- FP16 uses 16-bit.
- INT8 uses 8-bit.
- INT4 uses 4-bit.

Benefits:

- Lower memory.
- Faster inference.
- Cheaper deployment.

Tradeoff:

- Too much quantization can reduce quality.

---

## 25. Evaluation Of LLMs

LLM evaluation is hard because outputs are open-ended.

Evaluate:

- Correctness.
- Helpfulness.
- Factuality.
- Safety.
- Robustness.
- Instruction following.
- Latency.
- Cost.

### 25.1 Automatic Metrics

Perplexity:

- Measures how well model predicts text.
- Lower is better.

BLEU/ROUGE:

- Used in translation/summarization.
- Limited because they depend on exact word overlap.

Accuracy/F1:

- Useful for classification-style tasks.

### 25.2 Human Evaluation

Humans compare answers for:

- Accuracy.
- Clarity.
- Helpfulness.
- Safety.

### 25.3 LLM-as-Judge

Another model grades outputs.

Useful but risky:

- Judge model may be biased.
- Needs calibration with human labels.

### 25.4 RAG Evaluation

Evaluate:

- Retrieval precision.
- Retrieval recall.
- Faithfulness to retrieved context.
- Answer correctness.
- Citation quality.

---

## 26. Prompting

Prompting is giving instructions and context to guide the model.

Good prompt structure:

```text
Role / task
Context
Constraints
Output format
Examples if needed
Question
```

### 26.1 Zero-Shot

Ask directly without examples.

### 26.2 Few-Shot

Provide examples.

### 26.3 Chain-of-Thought Style

Ask model to reason step by step.

Practical caution:

- For final user-facing answers, ask for concise reasoning or explanation.
- Do not assume generated reasoning is always faithful.

### 26.4 Structured Output

Ask for JSON or a schema when you need machine-readable output.

---

## 27. Agents And Tool Use

An LLM by itself predicts text.

With tools, it can:

- Search documents.
- Run code.
- Query databases.
- Call APIs.
- Use calculators.
- Edit files.

Agent loop:

1. Understand goal.
2. Plan.
3. Choose tool.
4. Observe result.
5. Continue or answer.

Risks:

- Wrong tool calls.
- Security issues.
- Prompt injection.
- Infinite loops.
- Poor verification.

Good agent systems need:

- Tool permissions.
- Sandboxing.
- Logging.
- Validation.
- Human approval for risky actions.

---

## 28. Safety And Alignment

LLMs can produce harmful or wrong outputs if not controlled.

Concerns:

- Hallucination.
- Bias.
- Privacy leakage.
- Unsafe instructions.
- Misinformation.
- Prompt injection.
- Data poisoning.

Mitigations:

- Safety fine-tuning.
- Retrieval grounding.
- Refusal policies.
- Content filters.
- Human review.
- Monitoring.
- Red-teaming.

---

## 29. AI Images And Videos

Modern image and video generation is often based on diffusion models.

### 29.1 What Is A Diffusion Model?

A diffusion model learns to remove noise.

Training:

1. Start with a real image.
2. Add noise step by step.
3. Train model to predict/remove noise.

Generation:

1. Start from random noise.
2. Denoise step by step.
3. End with a realistic image.

### 29.2 Text-To-Image

Text prompt:

```text
a red car on a mountain road
```

The text is encoded into vectors.

The diffusion model uses those vectors as guidance while denoising.

### 29.3 Latent Diffusion

Instead of denoising full pixel images, latent diffusion works in compressed latent space.

Benefits:

- Faster.
- Less memory.

High-level flow:

```text
text prompt -> text encoder
noise in latent space -> denoising U-Net/transformer
latent -> decoder -> image
```

### 29.4 Classifier-Free Guidance

Guidance controls how strongly the image follows the prompt.

Higher guidance:

- More prompt adherence.
- May reduce naturalness.

Lower guidance:

- More natural/random.
- May ignore prompt.

### 29.5 AI Video Generation

Video generation is harder because frames must be:

- Visually good.
- Temporally consistent.
- Physically plausible.

Methods:

- Generate video directly with spatiotemporal model.
- Generate keyframes then interpolate.
- Use diffusion over space and time.
- Use transformer-based video models.

Challenges:

- Consistent objects across frames.
- Motion realism.
- Long duration.
- High compute cost.

---

## 30. Multimodal Models

Multimodal models handle more than text.

Modalities:

- Text.
- Image.
- Audio.
- Video.
- Code.
- Actions/tools.

Vision-language model:

- Takes image and text.
- Answers questions about image.

Speech model:

- Converts audio to text or responds with speech.

Multimodal LLM idea:

- Convert each modality into tokens or embeddings.
- Process with shared or connected model.

---

## 31. LLM System Design

For real applications, the model is only one part.

A production LLM app may include:

- User interface.
- Prompt templates.
- Retrieval system.
- Vector database.
- Model/API.
- Tool calling.
- Guardrails.
- Evaluation.
- Logging.
- Monitoring.
- Feedback loop.

### 31.1 Common LLM App Patterns

Chatbot:

- System prompt.
- Conversation memory.
- Safety filters.

Document question answering:

- RAG pipeline.
- Source citations.

Code assistant:

- Repository indexing.
- Tool execution.
- Tests.

Customer support:

- Knowledge base retrieval.
- Escalation to human.

Data analyst assistant:

- SQL generation.
- Chart generation.
- Query verification.

---

## 32. Common Failure Modes

Hallucination:

- Model makes up plausible but false information.

Prompt injection:

- User/document tries to override system instructions.

Context loss:

- Important information is outside context window or ignored.

Retrieval failure:

- RAG retrieves wrong documents.

Overconfidence:

- Model gives confident answer despite uncertainty.

Format failure:

- Model does not follow required JSON/schema.

Latency/cost problem:

- Model is too slow or expensive.

---

## 33. Practical LLM Workflow

For an LLM project:

1. Define task.
2. Decide if simple rules/search are enough.
3. Pick model.
4. Create evaluation set.
5. Build baseline prompt.
6. Add RAG if knowledge is needed.
7. Add tools if actions/calculation are needed.
8. Test failures.
9. Add guardrails.
10. Measure latency and cost.
11. Deploy.
12. Monitor.

Best habit:

- Build evaluation before heavy optimization.

---

## 34. Important Formulas

Neuron:

```text
z = w.x + b
a = activation(z)
```

Gradient descent:

```text
w = w - learning_rate * dL/dw
```

Softmax:

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

Cross entropy:

```text
L = -sum y_i * log(p_i)
```

Scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Residual connection:

```text
y = x + layer(x)
```

---

## 35. Interview Questions And Answers

### What is an LLM?

An LLM is a large transformer-based language model trained on huge text/code data to predict the next token. Because of this training, it learns grammar, facts, reasoning patterns, and language generation behavior.

### What is a token?

A token is a piece of text used by the model. It can be a word, part of a word, character, punctuation, or byte-level unit.

### What is an embedding?

An embedding is a vector representation of a token. It converts discrete text into numbers the neural network can process.

### What is attention?

Attention lets each token focus on relevant tokens in the context. It computes similarity between queries and keys, then uses the resulting weights to combine values.

### What are Q, K, and V?

Query means what the token is looking for. Key means what each token offers. Value is the information passed forward.

### Why do we divide by sqrt(d_k)?

To prevent dot products from becoming too large, which would make softmax too sharp and gradients unstable.

### What is self-attention?

Self-attention means tokens attend to other tokens in the same sequence.

### What is masked attention?

Masked attention blocks future tokens so the model can only use past context during generation.

### What is multi-head attention?

It runs multiple attention mechanisms in parallel so the model can learn different relationships.

### Why do transformers need positional encoding?

Attention alone has no natural sense of token order, so positional information is added.

### What is the difference between encoder-only, decoder-only, and encoder-decoder models?

Encoder-only models are good for understanding tasks. Decoder-only models are good for generation. Encoder-decoder models are good for sequence-to-sequence tasks like translation.

### What is pretraining?

Pretraining is training the model on huge data to learn general language patterns, usually by predicting tokens.

### What is fine-tuning?

Fine-tuning adapts a pretrained model to a specific task, style, or domain.

### What is RAG?

Retrieval augmented generation retrieves relevant documents and gives them to the model as context before answering.

### Fine-tuning vs RAG?

Use RAG for external or changing knowledge. Use fine-tuning for behavior, style, or task format.

### What is hallucination?

Hallucination is when a model generates plausible but false information.

### How can hallucination be reduced?

Use retrieval, tools, citations, better prompting, evaluation, refusal behavior, and human review for high-stakes tasks.

### What is KV cache?

KV cache stores previous attention keys and values during generation to avoid recomputing them, making inference faster.

### What is temperature?

Temperature controls randomness during token sampling. Lower is more deterministic; higher is more creative.

### What is quantization?

Quantization reduces number precision to make models smaller and faster, with possible quality tradeoffs.

### What is LoRA?

LoRA is a parameter-efficient fine-tuning method that freezes the base model and trains small low-rank adapter matrices.

### How do diffusion models generate images?

They start from noise and repeatedly denoise it, guided by the text prompt, until an image appears.

---

## 36. Final Summary

The core of LLMs:

```text
text -> tokens -> embeddings -> transformer blocks -> next-token probabilities
```

The core of training:

```text
predict next token -> calculate loss -> backpropagate gradients -> update weights
```

The core of attention:

```text
queries compare with keys, then weights combine values
```

The core of practical LLM apps:

```text
prompt + retrieval + tools + model + evaluation + monitoring
```

The core of AI images/videos:

```text
start with noise, denoise step by step, guide with text or other conditions
```

If you can explain these five lines clearly, you can answer most beginner-to-intermediate LLM interview questions.

