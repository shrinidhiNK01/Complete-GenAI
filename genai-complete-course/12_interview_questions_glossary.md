# 12. Interview Questions and Glossary

## Beginner Questions

### What is Generative AI?

Generative AI creates new content such as text, image, audio, video, or code. It learns patterns from training data and generates new outputs that follow those patterns.

### What is the difference between AI, ML, DL, and GenAI?

AI is the broad goal of intelligent machines. ML is AI that learns from data. DL is ML with deep neural networks. GenAI is usually DL that creates new content.

### What is a token?

A token is a piece of text used by a model. It can be a word, part of a word, punctuation, or symbol.

### What is hallucination?

Hallucination is when a model gives an answer that sounds confident but is false or unsupported.

## LLM Questions

### Why did Transformers replace RNNs/LSTMs for many NLP tasks?

Transformers handle long-range dependencies better, train in parallel, and scale very well with data and compute. RNNs process tokens sequentially, which makes them slower and harder to scale.

### What is attention?

Attention lets each token decide which other tokens are important for understanding it.

### What is pretraining?

Pretraining is the large-scale training stage where a model learns general patterns from massive data.

### What is instruction tuning?

Instruction tuning teaches the model to follow human instructions using instruction-answer examples.

## RAG Questions

### What is RAG?

RAG retrieves relevant external information and gives it to an LLM so the answer is grounded in that information.

### Why use RAG instead of fine-tuning?

Use RAG when knowledge changes often or lives in private documents. Fine-tuning is better for style, format, and repeated behavior.

### What are embeddings?

Embeddings are numeric vectors that represent semantic meaning.

### What is vector search?

Vector search finds items with embeddings similar to the query embedding.

### What is chunking?

Chunking splits documents into smaller pieces for retrieval.

### What makes chunking hard?

Chunks must be small enough to retrieve precisely but large enough to preserve meaning.

### What is reranking?

Reranking reorders retrieved candidate chunks using a more accurate model.

### What is hybrid search?

Hybrid search combines keyword search and vector search.

### What is HyDE?

HyDE generates a hypothetical answer/document, embeds it, and uses that embedding for retrieval.

### What is the lost-in-the-middle problem?

LLMs may ignore important information placed in the middle of a long context.

## Agent Questions

### What is an AI agent?

An AI agent uses an LLM to plan, call tools, observe results, and continue until it completes a goal.

### What is ReAct?

ReAct combines reasoning and acting. The agent reasons, chooses a tool, observes the result, and repeats.

### What is LangGraph?

LangGraph is a framework for building stateful, graph-based agent workflows.

### What is human-in-the-loop?

Human-in-the-loop means the system pauses for human review or approval before continuing.

## Fine-Tuning Questions

### What is fine-tuning?

Fine-tuning updates model weights using a task-specific dataset.

### What is LoRA?

LoRA is a parameter-efficient fine-tuning method that trains small adapter matrices instead of all model weights.

### What is QLoRA?

QLoRA combines quantization with LoRA to reduce GPU memory usage.

### What is DPO?

DPO trains a model from chosen/rejected preference pairs without a separate reward model.

### What is RLHF?

RLHF uses human feedback to train a reward model and optimize the model toward preferred behavior.

## LLMOps Questions

### What should you log in an LLM app?

Log request IDs, prompt version, model, retrieval IDs, latency, token usage, errors, and safety decisions. Do not log secrets.

### How do you evaluate RAG?

Evaluate retrieval relevance, answer correctness, faithfulness, citation accuracy, latency, and cost.

### How do you reduce hallucination?

Use RAG, citations, strict prompts, answer verification, abstention rules, evals, and guardrails.

### How do you deploy an LLM app?

Common flow: FastAPI backend, Docker image, cloud service such as ECS/Fargate, EKS, Azure App Service, AKS, or GCP Cloud Run, plus CI/CD and monitoring.

## Glossary

| Term | Simple Meaning |
| --- | --- |
| AI | Machines doing intelligent tasks |
| ML | Learning patterns from data |
| DL | Neural networks with many layers |
| GenAI | AI that creates new content |
| NLP | AI for human language |
| Token | Piece of text processed by model |
| Embedding | Vector representing meaning |
| Vector DB | Database for similarity search |
| RAG | Retrieval plus generation |
| Chunk | Small document piece |
| Retriever | Component that finds relevant chunks |
| Reranker | Component that reorders candidates |
| BM25 | Keyword search ranking algorithm |
| Hybrid Search | Keyword plus vector search |
| HyDE | Hypothetical document embedding |
| Agent | LLM system that uses tools and steps |
| Tool Calling | Model requests a function/API |
| LangGraph | Graph framework for agent workflows |
| Fine-tuning | Training a model further on custom data |
| LoRA | Lightweight fine-tuning adapters |
| QLoRA | Quantized LoRA fine-tuning |
| RLHF | Reinforcement learning from human feedback |
| DPO | Direct preference optimization |
| Distillation | Train small model from large model |
| Quantization | Lower precision to reduce memory |
| Guardrail | Safety/control layer |
| LLMOps | Production operations for LLM apps |

## Revision Checklist

- Explain AI vs ML vs DL vs GenAI.
- Draw Transformer architecture.
- Explain tokenization.
- Explain embeddings with cosine similarity.
- Build a basic RAG pipeline.
- Explain chunking tradeoffs.
- Explain hybrid search and reranking.
- Explain LangChain vs LlamaIndex.
- Build a basic tool-calling agent.
- Explain memory types.
- Explain LangGraph nodes, edges, and state.
- Explain RAG vs fine-tuning vs agents.
- Explain LoRA and QLoRA.
- Explain RLHF and DPO.
- Deploy a FastAPI LLM app with Docker.
- Add tests and evals.
- Add guardrails and monitoring.

