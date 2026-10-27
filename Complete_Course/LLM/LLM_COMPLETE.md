# Large Language Models: The Complete Masterclass (Beginner to Advanced)

Welcome to the ultimate guide to Large Language Models (LLMs). This document explains every major LLM concept, training phase, PEFT technique, quantization method, prompting pattern, and evaluation benchmark in simple English, followed by step-by-step implementation notes.

---

## Table of Contents
1. **Introduction to Large Language Models**
2. **Tokenization (BPE, WordPiece, SentencePiece)**
3. **Training Phases: Pre-training, SFT, and RLHF/DPO**
4. **Parameter-Efficient Fine-Tuning: LoRA & QLoRA**
5. **Model Quantization (GPTQ, AWQ, GGUF)**
6. **Prompt Engineering Design Patterns**
7. **LLM Evaluation Benchmarks & LLM-as-a-Judge**

---

## 1. Introduction to Large Language Models

LLMs are massive neural networks based on the Transformer architecture, trained on trillions of words from the internet. They work by calculating probabilities over a vocabulary to repeatedly predict the next token.

### Architecture Categories
*   **Decoder-Only (Autoregressive):** Predicts the next word given previous words. Standard for text generation (e.g. GPT-4, Llama 3, Mistral, Gemini).
*   **Encoder-Only:** Learns bidirectional context (looks left and right). Used for classification, translation, and embeddings (e.g. BERT).
*   **Encoder-Decoder:** Encodes input sequence, then decodes output sequence. Used for summarization and translation (e.g. T5, BART).

---

## 2. Tokenization

Computers don't read words; they read numbers. A **Tokenizer** splits text into chunks called **Tokens** and maps them to integer IDs.

```text
Text: "unhappiness" ===> Tokenizer ===> ["un", "happi", "ness"] ===> IDs: [421, 9872, 102]
```

### Why not use characters or whole words?
*   *Characters:* Too fine-grained. The model has to spend too much computation learning how to spell words.
*   *Whole Words:* Vocabulary becomes too large (millions of words). The model cannot handle new words (Out-Of-Vocabulary).

### Subword Tokenization Algorithms

| Algorithm | How it works | Used by |
| :--- | :--- | :--- |
| **Byte Pair Encoding (BPE)** | Starts with individual characters. Counts the most frequent adjacent byte/character pairs and merges them. Repeats until vocabulary size is reached. | GPT, Llama |
| **WordPiece** | Similar to BPE, but instead of merging the most *frequent* pair, it merges the pair that maximizes the *likelihood* of the training data according to a language model. | BERT |
| **SentencePiece** | Treats the input as a raw byte stream (including spaces, represented as `_`), meaning it doesn't require pre-tokenizing text by spaces first. Ideal for multilingual datasets. | T5, Llama |

*See a complete BPE Tokenizer built from scratch in: `implementations/custom_tokenizer.py`*

---

## 3. Training Phases: From Raw Text to Helpful Assistant

Training a production assistant LLM involves three sequential phases:

```text
[ Raw Internet Text ] ===> Phase 1: Pre-training (Base Model)
                                |
[ Instruction Pairs ] ===> Phase 2: Supervised Fine-Tuning (SFT Model)
                                |
[ Chosen/Rejected Pairs ] ===> Phase 3: Alignment (RLHF / DPO Model) ===> Helpful Assistant
```

### Phase 1: Pre-training (The Base Model)
*   **Goal:** Learn grammar, facts about the world, and basic reasoning.
*   **Dataset:** Trillions of tokens of raw text (Wikipedia, books, crawled websites).
*   **Objective:** Next-token prediction (Autoregressive Language Modeling).
*   **Result:** A **Base Model** (e.g., Llama-3-Base). It is excellent at completing patterns but makes a poor chatbot. (If you ask: *"Write a poem about dogs"*, it might respond with: *"Write a poem about cats, write a poem about birds"* because it completes the pattern of listing writing prompts).

### Phase 2: Supervised Fine-Tuning (SFT)
*   **Goal:** Teach the model how to act like an assistant and follow instructions.
*   **Dataset:** Thousands of high-quality, curated prompt-response pairs.
    *   *Prompt:* *"Explain gravity in simple terms."*
    *   *Response:* *"Gravity is an invisible force that pulls objects toward each other..."*
*   **Result:** An **Instruction-Tuned Model** (e.g., Llama-3-Instruct). It knows how to answer questions.

### Phase 3: Alignment (RLHF & DPO)
*   **Goal:** Ensure the model is helpful, honest, and harmless. It aligns the model's behavior with human values.
*   **Method 1: RLHF (Reinforcement Learning from Human Feedback):**
    1.  Collect comparison data (humans rate model outputs).
    2.  Train a separate **Reward Model** to predict how much a human would like an output.
    3.  Optimize the SFT model using **PPO (Proximal Policy Optimization)** reinforcement learning to generate text that maximizes reward.
*   **Method 2: DPO (Direct Preference Optimization):**
    *   Instead of training a complex reward model and using reinforcement learning, DPO mathematically updates the SFT model directly using preference pairs (a prompt, a chosen response, and a rejected response). It is faster, more stable, and the standard in modern pipelines.

---

## 4. Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning requires updating all billions of parameters of an LLM, which takes huge amounts of GPU memory. PEFT solves this.

### LoRA: Low-Rank Adaptation
During fine-tuning, the change in weights ($\Delta W$) can be represented as a lower-rank matrix without losing much information.

Instead of updating the original weight matrix $W_0$ (shape $d \times d$), LoRA freezes it and adds two small matrices, $A$ and $B$, in parallel.

$$\Delta W = B \cdot A$$

```text
Input x ===> [ Original Weight Matrix W0 (FREEZING) ] ===> Output y0 \
        ===> [ Small Matrix A ] ===> [ Small Matrix B ]  ===> Output y1 / ==> y = y0 + y1
```

If $d = 4096$ and we choose rank $r = 8$:
*   Original $W_0$ parameters: $4096 \times 4096 \approx 16.7$ million parameters.
*   LoRA parameters: $A$ ($4096 \times 8$) + $B$ ($8 \times 4096$) $\approx 65,536$ parameters.
*   **Reduction:** 99.6% fewer parameters to train!

### QLoRA: Quantized LoRA
QLoRA takes LoRA further to fit even larger models on consumer GPUs:
1.  Quantizes the frozen base model to **4-bit NormalFloat (NF4)** (a data type optimized for normally distributed weights).
2.  Applies double quantization to compress the scaling factors.
3.  Performs fine-tuning using 16-bit LoRA adapters on top of the 4-bit base model.

*See fine-tuning script in: `implementations/finetune_lora.py`*

---

## 5. Model Quantization

Quantization compresses models by reducing the precision of their weights (e.g., converting 16-bit floating-point numbers to 8-bit or 4-bit integers).

| Quantization Type | How it works | Best Use Case |
| :--- | :--- | :--- |
| **GGUF** | A file format developed by llama.cpp. Optimized for running models on consumer CPUs and Apple Silicon by offloading layers to GPUs. | Running local LLMs on laptops. |
| **GPTQ** | Post-Training Quantization. Calibrates weights on a small dataset, adjusting remaining weights to compensate for quantization errors. | GPU inference. |
| **AWQ (Activation-aware Quantization)** | Protects the 1% "salient" weights that are most important for activations, while quantizing the other 99%. Reduces loss of accuracy. | High-throughput GPU inference. |

---

## 6. Prompt Engineering Design Patterns

### Few-Shot Prompting
Provide a few examples in the prompt to show the model the desired output format.

```text
Review: "This movie is amazing!" -> Sentiment: Positive
Review: "I fell asleep halfway through." -> Sentiment: Negative
Review: "It was okay, but a bit long." -> Sentiment:
```

### Chain-of-Thought (CoT) Prompting
Instruct the model to break down its reasoning step-by-step before outputting the final answer. This significantly improves performance on math, logic, and coding tasks.

```text
Prompt: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? Let's think step-by-step."
```

### ReAct Framework (Reasoning + Acting)
Allows agents to solve tasks by alternating between reasoning ("Thought") and using tools ("Action" followed by "Observation").

```text
Thought: I need to find the population of Paris in 2026.
Action: Search[population of Paris 2026]
Observation: Paris population is estimated at 2.1 million.
Thought: Now I have the answer. I will respond to the user.
```

---

## 7. LLM Evaluation Benchmarks

Evaluating generative text is hard because there are many ways to write a correct answer.

### Common Benchmarks
*   **MMLU (Massive Multitask Language Understanding):** Multiple-choice questions across subjects (humanities, sciences, law, math) to test academic knowledge.
*   **GSM8K:** Grade-school math word problems to test reasoning.
*   **HumanEval:** Python coding challenges. The model's code is executed to see if it passes unit tests (Pass@1 metric).

### LLM-as-a-Judge
Instead of using fixed rules (like BLEU/ROUGE) or hiring expensive humans, we use a powerful LLM (like GPT-4) to evaluate the outputs of other models.
1.  **Direct Scoring:** Ask the judge model to rate a response from 1 to 10 based on criteria.
2.  **Pairwise Comparison:** Present the judge model with a prompt and two responses (from Model A and Model B) and ask it to choose the better response and explain why.
    *   *Watch out for Bias:* LLMs tend to prefer longer answers, and they have "position bias" (favoring whichever answer is presented first).
