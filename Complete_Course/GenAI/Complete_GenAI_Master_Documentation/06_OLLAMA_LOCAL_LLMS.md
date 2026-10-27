# Ollama and Local LLMs Complete Notes

## 1. What Is Ollama?

Ollama is a tool for running LLMs locally.

It provides:

- Command line interface
- Local REST API
- Model management
- Embedding models
- Python library
- JavaScript library

Default local API base URL:

```text
http://localhost:11434/api
```

## 2. Why Run Models Locally?

Benefits:

- Privacy
- Offline usage
- Lower API cost
- Fast prototyping
- Control over models
- Useful for learning

Limitations:

- Needs enough RAM/VRAM.
- Smaller local models may be weaker than top cloud models.
- Setup and performance depend on hardware.
- Long context can be slower.

## 3. Basic Ollama Commands

Install a model:

```bash
ollama pull llama3.2
```

Run a model:

```bash
ollama run llama3.2
```

List models:

```bash
ollama list
```

Remove model:

```bash
ollama rm llama3.2
```

## 4. Ollama Generate API

Example:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Explain RAG in simple English."
}'
```

## 5. Ollama Chat API

Chat is used for multi-message conversations.

Concept:

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain LangGraph."}
  ]
}
```

## 6. Ollama Embeddings

Embeddings convert text into vectors for semantic search and RAG.

Example model names may include:

- `embeddinggemma`
- `all-minilm`
- `qwen3-embedding`

API concept:

```bash
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
```

## 7. Ollama With LangChain

You can use local Ollama models inside LangChain.

Example:

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2", temperature=0)
response = model.invoke("Explain embeddings in simple English.")
print(response.content)
```

Embeddings:

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="embeddinggemma")
vector = embeddings.embed_query("What is RAG?")
print(len(vector))
```

## 8. Local RAG With Ollama

Architecture:

```text
Local docs -> splitter -> Ollama embeddings -> local vector DB -> Ollama chat model -> answer
```

Good for:

- Personal notes
- Offline study assistant
- Private documents

## 9. Choosing Local Models

Consider:

- Task type: chat, code, embeddings
- Model size
- RAM/VRAM
- Context length
- Tool support
- Structured output support
- Speed

General guidance:

- Small models: faster, less accurate.
- Larger models: slower, better reasoning.
- Embedding models: use dedicated embedding models, not normal chat models.

## 10. Quantization

Quantization reduces model size by using lower precision numbers.

Benefits:

- Less memory
- Faster on limited hardware

Tradeoff:

- Some quality loss

## 11. Local LLM Production Concerns

For production local deployments:

- Monitor latency.
- Limit concurrent requests.
- Use appropriate hardware.
- Validate outputs.
- Add fallback models if needed.
- Keep models updated.
- Protect local API access.

## 12. Ollama Interview Answer

Say this:

```text
Ollama lets us run open LLMs locally through a CLI and local API. It is useful for privacy, offline work, and low-cost prototyping. It can run chat models and embedding models, so we can build local RAG systems. The main tradeoffs are hardware requirements, speed, and model quality compared with strong cloud models.
```
