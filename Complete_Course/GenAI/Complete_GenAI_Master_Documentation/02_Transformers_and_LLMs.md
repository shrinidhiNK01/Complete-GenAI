# Transformers and Large Language Models (LLMs)

This document provides a comprehensive, beginner-to-advanced walkthrough of the Transformer architecture, LLM training lifecycles, fine-tuning protocols, quantization, and inference optimizations.

---

## 1. The Transformer Architecture

Before 2017, sequential models like RNNs and LSTMs read sentences word-by-word. This was slow because it could not be parallelized on GPUs, and models struggled to remember long-term dependencies. 

In 2017, Vaswani et al. introduced the **Transformer** in the paper *"Attention Is All You Need"*, discarding recurrence entirely and relying on **Self-Attention** to process all tokens at once.

```text
Input Tokens ===> [ Embedding + Positional Encoding ]
                           │
                           ▼
                  [ Multi-Head Attention ] <----\
                           │                     |  (Repeated N times)
                           ▼                     |
                  [ LayerNorm & Residual ]       |
                           │                     |
                           ▼                     |
                  [ Feed-Forward Network ] ------/
                           │
                           ▼
                       [ Logits ] ===> Softmax ===> Next Token
```

### Key Components

#### 1. Tokenization & Embeddings
*   **Tokenization:** Converts raw text strings into a list of integer IDs representing subwords.
*   **Embeddings:** A lookup table that converts each integer token ID into a high-dimensional vector (e.g., 4096 dimensions) that represents its initial semantic meaning.

#### 2. Positional Encoding
Because Transformers process all tokens simultaneously, they do not inherently know the order of words. We solve this by adding a **Positional Encoding** vector to the input embeddings. This is calculated using sine and cosine waves of different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

This injects a unique spatial coordinate for each word in the sequence.

#### 3. Self-Attention (Q, K, V)
Self-attention computes how much each word in a sentence should focus on (attend to) every other word.
*   **Query ($Q$):** "What am I looking for?" (Current word).
*   **Key ($K$):** "What identifiers do other words have?" (To match against the Query).
*   **Value ($V$):** "What content do those words actually contain?"

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

*   **$QK^T$:** Dot-product multiplication of queries and keys to get a raw attention score.
*   **$\sqrt{d_k}$ Scaling:** Divides by the square root of the key dimension to prevent scores from growing too large, which would push the softmax function into regions with tiny gradients.
*   **Softmax:** Converts scores to probabilities (weights).
*   **Multiply by $V$:** Multiplies probabilities by the values, keeping important context and discarding noise.

#### 4. Multi-Head Attention
Instead of calculating attention once, the model splits $Q, K, V$ into $H$ separate heads, calculates attention in parallel, and concatenates the outputs. This allows different heads to focus on different linguistic aspects (e.g., one head handles verbs, another matches pronouns to nouns).

#### 5. Feed-Forward Network (FFN)
Applied to each position separately. It consists of two linear transformations with a non-linear activation (usually GELU or SwiGLU) in between:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

---

## 2. LLM Training Lifecycle

Creating an assistant model consists of three sequential phases:

```text
Phase 1: Pre-training  ===> Phase 2: SFT (Supervised Fine-Tuning) ===> Phase 3: Alignment (DPO/RLHF)
 (Unsupervised, Base)        (Supervised, Instruction Following)          (Human Value Tuning)
```

### Phase 1: Pre-training (Base Models)
*   **Training Objective:** Predict the next token given context (Autoregressive Language Modeling).
*   **Dataset:** Terabytes of raw web text (crawls, books, code).
*   **Output:** **Base Model** (e.g., Llama-3-Base). It knows grammar and facts but completes text patterns rather than answering questions.

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Training Objective:** Teach the model to follow instructions.
*   **Dataset:** Prompt-response pairs curated by humans (e.g. Q: *"Translate this"* $\to$ A: *"French translation"*).
*   **Output:** **SFT/Instruction Model** (e.g., Llama-3-Instruct). It knows how to answer questions.

### Phase 3: Alignment (RLHF & DPO)
*   **Goal:** Make the model helpful, honest, and safe.
*   **RLHF (Reinforcement Learning from Human Feedback):** Uses human comparison data to train a **Reward Model**. The SFT model is then optimized using PPO reinforcement learning to generate responses that score highly on the reward model.
*   **DPO (Direct Preference Optimization):** Skips the reward model entirely. It optimizes the SFT model directly using a loss function based on preferred (chosen) and non-preferred (rejected) outputs, ensuring higher stability and speed.

---

## 3. Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning requires updating all parameters of an LLM, demanding massive GPU memory. PEFT techniques reduce this footprint.

### LoRA (Low-Rank Adaptation)
LoRA freezes the pre-trained model weights $W_0$ (shape $d \times k$) and injects trainable rank decomposition matrices $A$ and $B$ in parallel.

$$\Delta W = B \times A \quad \text{where } A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

*   During training, only $A$ and $B$ are updated.
*   For inference, the product $B \times A$ is added back to the base weight $W_0$, resulting in zero extra inference latency.
*   **Saves:** Up to 99% of trainable parameters and significant VRAM.

### QLoRA
QLoRA optimizes LoRA by:
1.  Quantizing the frozen base model to a specialized **4-bit NormalFloat (NF4)** format.
2.  Using **Double Quantization** to compress the quantization constants.
3.  Fine-tuning using 16-bit LoRA adapter weights, enabling large models (e.g. 70B parameters) to be fine-tuned on a single workstation GPU.

---

## 4. Model Quantization

Quantization reduces memory requirements by storing weights in lower-precision representations.

*   **FP16 / BF16 (16-bit float):** Standard precision for model training/inference (2 bytes per parameter).
*   **INT8 (8-bit integer):** Halves footprint (1 byte per parameter).
*   **INT4 / NF4 (4-bit representation):** Compresses parameters to 0.5 bytes, allowing massive models to fit in consumer GPUs.

### Quantization Formats
*   **GGUF:** Designed for running LLMs on consumer CPUs and Apple Silicon (using llama.cpp). It bundles the weights and vocabulary into a single file and supports hybrid CPU/GPU offloading.
*   **GPTQ:** Post-training quantization method that optimizes weights on calibration datasets to minimize accuracy loss in 4-bit/3-bit. Excellent for GPU servers.
*   **AWQ (Activation-aware Weight Quantization):** Protects the 1% most important weights (based on layer activations) by keeping them in high precision, while quantizing the remaining 99%.

---

## 5. Inference Optimization

Running LLM inference at scale requires techniques to combat GPU memory bandwidth bottlenecks:

### KV Cache (Key-Value Caching)
*   **The Problem:** In autoregressive generation, to generate token $t$, the model must recompute key and value representations for all previous tokens $1 \dots t-1$. This causes huge redundant matrix multiplications.
*   **The Solution:** Save (cache) the $K$ and $V$ matrices for all tokens in memory, so we only compute the projections for the newly generated token at each step. This drastically reduces latency at the expense of GPU memory.

### FlashAttention
*   **The Problem:** Traditional attention calculations scale quadratically ($O(N^2)$) with sequence length and require writing huge intermediate attention matrix grids to slow High Bandwidth Memory (HBM).
*   **The Solution:** flashAttention computes attention in blocks using GPU SRAM (ultra-fast cache) without writing the intermediate $N \times N$ matrix back to HBM, speeding up training and inference by 2x to 4x.
