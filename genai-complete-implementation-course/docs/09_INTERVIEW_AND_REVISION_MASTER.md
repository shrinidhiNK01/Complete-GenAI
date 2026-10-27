# 09. Interview and Revision Master

## Core Interview Questions

### What is Generative AI?

Generative AI is AI that creates new content such as text, image, audio, video, or code.

### What is RAG?

RAG retrieves relevant information from an external source and gives it to a model so the model can answer with grounded context.

### Why use RAG?

Use RAG for private data, fresh data, source citations, and hallucination reduction.

### What is an embedding?

An embedding is a numeric vector that represents meaning.

### What is BM25?

BM25 is a keyword ranking algorithm. It is strong for exact terms, names, IDs, and rare words.

### What is hybrid search?

Hybrid search combines vector search and keyword search.

### What is reranking?

Reranking takes candidate results from retrieval and sorts them more accurately.

### What is RRF?

RRF means reciprocal rank fusion. It merges multiple ranked lists.

### What is HyDE?

HyDE creates a hypothetical answer/document, embeds it, and retrieves documents similar to it.

### What is parent document retrieval?

It indexes small child chunks but returns larger parent sections for context.

### What is lost in the middle?

LLMs may ignore important context placed in the middle of a long prompt.

### What is an agent?

An agent uses an LLM to choose steps, call tools, observe results, and continue toward a goal.

### What is ReAct?

ReAct combines reasoning and acting in a loop.

### What is LangGraph?

LangGraph builds stateful graph workflows for agents.

### What is fine-tuning?

Fine-tuning updates model weights using task-specific data.

### What is LoRA?

LoRA trains small adapter matrices instead of all model weights.

### What is QLoRA?

QLoRA combines quantization with LoRA to reduce memory usage.

### What is RLHF?

RLHF uses human feedback to train a reward model and optimize the model.

### What is DPO?

DPO trains from chosen/rejected answer pairs directly.

### What is LLMOps?

LLMOps is the practice of testing, deploying, monitoring, evaluating, and improving LLM applications.

## Practical Interview Task: Build RAG

You should be able to explain:

1. Load documents.
2. Clean text.
3. Chunk text.
4. Create embeddings.
5. Store in vector DB.
6. Retrieve top chunks.
7. Rerank.
8. Build prompt.
9. Generate answer.
10. Cite sources.
11. Evaluate answer.
12. Deploy API.

## Practical Interview Task: Debug Bad RAG

Question:

The system answers incorrectly. What do you check?

Answer:

1. Was the source document ingested?
2. Was text extracted correctly?
3. Was chunking too small or too large?
4. Did retrieval return the right chunk?
5. Did reranking drop the right chunk?
6. Did prompt include the context?
7. Did model ignore the context?
8. Is there a prompt injection?
9. Is metadata filtering wrong?
10. Are evals covering this case?

## Revision Checklist

- Explain AI vs ML vs DL vs GenAI.
- Explain RNN, LSTM, GRU.
- Explain attention and Transformer.
- Explain pretraining and instruction tuning.
- Explain embeddings and cosine similarity.
- Build RAG from scratch.
- Explain BM25.
- Explain hybrid retrieval.
- Explain reranking.
- Explain RAG fusion and HyDE.
- Explain parent document and sentence window retrieval.
- Explain corrective RAG, self-RAG, agentic RAG.
- Build a tool-using agent.
- Explain memory types.
- Explain multi-agent patterns.
- Explain fine-tuning vs RAG vs agents.
- Explain LoRA and QLoRA.
- Explain RLHF and DPO.
- Explain Docker deployment.
- Explain CI/CD.
- Explain guardrails and evals.

