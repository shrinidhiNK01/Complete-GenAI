# GenAI, ML, DL, and RAG Interview Master Preparation

This document serves as your final cheat sheet and preparation guide for interviews. It covers high-frequency questions, technical comparisons, architectural design problems, and debugging scenarios.

---

## 1. High-Frequency Interview Q&A

### Q1: What is the difference between BLEU, ROUGE, and FID?
*   **BLEU (Bilingual Evaluation Understudy):** Measures **Precision** of generated text against a human reference by checking how many n-grams in the generated text appear in the reference. (Used in translation).
*   **ROUGE (Recall-Oriented Understudy for Gisting):** Measures **Recall** by checking how many n-grams in the human reference appear in the generated text. (Used in summarization).
*   **FID (Fréchet Inception Distance):** Measures similarity of two image distributions. It feeds real and generated images through an Inception model, extracts intermediate features, and calculates their statistical distance. Lower is better.

### Q2: How do L1 and L2 regularization differ?
*   **L1 (Lasso):** Adds the absolute sum of weights ($\lambda \sum |w_i|$) to the loss. It shrinks weights to **exactly zero**, acting as a feature selector.
*   **L2 (Ridge):** Adds the squared sum of weights ($\lambda \sum w_i^2$) to the loss. It shrinks weights to be small but **never zero**, distributing weight impacts smoothly.

### Q3: What is the vanishing gradient problem, and how do you fix it?
*   **The Problem:** During backpropagation through deep networks or long RNN sequences, gradients are multiplied repeatedly by small weights, shrinking to zero. Early layers stop learning.
*   **Fixes:**
    1.  Use activation functions like **ReLU** or **Leaky ReLU** instead of Sigmoid/Tanh.
    2.  Use **Batch Normalization** or **Layer Normalization**.
    3.  Use architectures with **Residual Connections** (ResNet, Transformers) which allow gradients to flow directly back.
    4.  Use LSTM/GRU cells instead of vanilla RNNs.

---

## 2. Core Technical Comparisons

| Concept A | Concept B | Core Difference |
| :--- | :--- | :--- |
| **Sigmoid** | **Softmax** | Sigmoid outputs a single probability between 0 and 1 (binary classification). Softmax outputs a probability distribution summing to 1 across multiple classes (multi-class). |
| **Bagging** | **Boosting** | Bagging trains trees in parallel independently (reduces variance, e.g. Random Forest). Boosting trains trees sequentially, correcting errors of past trees (reduces bias, e.g. XGBoost). |
| **Bi-Encoder** | **Cross-Encoder** | Bi-encoders embed query and documents separately (fast vector search). Cross-encoders calculate attention between query and document together (slow but highly accurate re-ranking). |
| **RLHF** | **DPO** | RLHF requires training a separate reward model and using complex PPO reinforcement learning. DPO bypasses the reward model, updating weights directly using preference pairs. |
| **LoRA** | **Full Fine-Tuning** | Full updates all parameters. LoRA freezes base weights and trains small low-rank projection matrices, reducing parameter count by 99%+. |

---

## 3. Real-World Debugging & Troubleshooting

### Scenario 1: The RAG pipeline returns good sources but the LLM answers "I don't know".
*   **Root Cause:**
    1.  *Context Overflow:* The retrieved chunks are too long, pushing the answer out of the context window.
    2.  *Lost in the Middle:* LLMs tend to ignore information in the middle of long prompts.
    3.  *Strict System Prompts:* The system prompt might be overly strict, instructing the model to say "I don't know" if there isn't a 100% exact match.
*   **Fixes:**
    *   Optimize chunk sizes and use parent-child chunking.
    *   Place the most relevant chunks at the very beginning or end of the context prompt.
    *   Tune the system prompt constraints.

### Scenario 2: The neural network training loss stops decreasing and hovers at a high value.
*   **Root Cause:**
    1.  *Learning Rate:* If too high, the model is bouncing around the minimum. If too low, it's taking too long.
    2.  *Dying ReLU:* Many neurons have negative inputs and are outputting exactly zero, stopping weight updates.
    3.  *Bad Initialization:* All weights initialized to the same value (network is symmetric).
*   **Fixes:**
    *   Use learning rate schedulers or change optimizer to **Adam**.
    *   Switch to **Leaky ReLU** or **GELU**.
    *   Use Xavier or He initialization.

---

## 4. Production System Design Case Study

### Question: Design a RAG-based customer support system handling 1 million documents.

```text
Ingestion:
[ Docs ] ===> [ SentenceSplitter ] ===> [ Embedder ] ===> [ IVF-HNSW Index (Vector DB) ]

Inference:
Query ===> [ Query Expansion ] ===> Vector Search (Top 50) ===> [ Cross-Encoder Reranker ]
                                                                        │
                                                                   Keep Top 5
                                                                        │
                                                                        ▼
User <=== [ Streaming Response ] <=== [ LLM API ] <=== Inject Context into Prompt
```

#### Architectural Choices
1.  **Ingestion & Storage:**
    *   *Chunking:* Use **Parent-Child Chunking** (100-character child chunks for search, yielding 1000-character parent chunks for answer context).
    *   *Storage:* Store vectors in a database indexing with **HNSW** for sub-millisecond retrieval speeds.
2.  **Retrieval Pipeline:**
    *   *Hybrid Search:* Combine BM25 keyword search (for product numbers/SKUs) and Dense vector search, merged using **Reciprocal Rank Fusion (RRF)**.
    *   *Re-ranking:* Retrieve 50 candidate documents and run them through a local **Cross-Encoder** model to select the top 5 most relevant.
3.  **Serving & Latency:**
    *   *KV Cache:* Enable KV Caching to speed up text generation.
    *   *Streaming:* Use FastAPI `StreamingResponse` to stream tokens to the user as they are generated to reduce perceived latency.
