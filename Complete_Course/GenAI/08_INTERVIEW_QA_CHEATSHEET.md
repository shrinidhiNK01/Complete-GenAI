# Interview Q&A Cheatsheet

## 1. What Is Generative AI?

Generative AI creates new content such as text, code, images, summaries, and plans. It learns patterns from data and generates output from a prompt.

## 2. What Is an LLM?

An LLM is a Large Language Model trained to predict the next token. It can answer, summarize, translate, reason, and generate code, but it can hallucinate because it generates likely text rather than guaranteed truth.

## 3. What Is a Token?

A token is a small piece of text processed by the model. It may be a word, part of a word, punctuation, or symbol.

## 4. What Is a Context Window?

The context window is the maximum number of tokens the model can consider at once, including prompt, chat history, retrieved context, and output.

## 5. What Is Prompt Engineering?

Prompt engineering is writing clear instructions so the model performs the task correctly. A good prompt includes role, task, context, constraints, examples, and output format.

## 6. What Is LangChain?

LangChain is a framework for building LLM applications by composing models, prompts, output parsers, retrievers, tools, and agents.

## 7. What Are Runnables?

Runnables are LangChain components with a standard interface such as `invoke`, `batch`, and `stream`. They make it easy to compose chains using LCEL.

## 8. What Is LCEL?

LCEL means LangChain Expression Language. It lets you compose components with the pipe operator, for example `prompt | model | parser`.

## 9. What Is Structured Output?

Structured output makes the model return data in a fixed schema such as JSON or a Pydantic model. It is useful for applications that need reliable machine-readable output.

## 10. What Is RAG?

RAG means Retrieval-Augmented Generation. It retrieves relevant external documents and gives them to the LLM so the answer is grounded in provided context.

## 11. Why Use RAG?

Use RAG when the model needs private, recent, domain-specific, or large knowledge that is not available in the model's training data.

## 12. What Are Embeddings?

Embeddings are numeric vectors representing semantic meaning. Similar texts have vectors close to each other.

## 13. What Is a Vector Store?

A vector store stores embeddings and performs similarity search to retrieve relevant chunks.

## 14. What Is a Retriever?

A retriever takes a query and returns relevant documents. In LangChain it commonly returns `Document` objects.

## 15. What Is Chunking?

Chunking splits large documents into smaller pieces so retrieval can find focused context and the LLM can fit it into the prompt.

## 16. What Is Chunk Overlap?

Chunk overlap repeats some text between chunks to avoid losing context at boundaries.

## 17. What Is Hybrid Search?

Hybrid search combines semantic vector search with keyword search. It is useful when exact terms and meaning both matter.

## 18. What Is Reranking?

Reranking reorders retrieved documents using a stronger relevance model so the best chunks appear first.

## 19. What Is Corrective RAG?

Corrective RAG checks whether retrieved documents are relevant. If retrieval is poor, it rewrites the query or searches another source before answering.

## 20. What Is Self-RAG?

Self-RAG lets the system evaluate whether retrieval is needed, whether documents are relevant, and whether the final answer is supported.

## 21. What Is an Agent?

An agent is an LLM system that can decide actions, call tools, observe results, and continue until it produces a final answer.

## 22. Agent vs Chain?

A chain follows fixed steps. An agent decides steps dynamically.

## 23. When Should You Not Use an Agent?

Do not use an agent for simple fixed tasks like summarization, classification, or basic RAG. Use a chain or graph instead.

## 24. What Are Tools?

Tools are functions the model can call, such as search, calculator, database query, or API call.

## 25. What Is Tool Calling?

Tool calling is when the model returns a structured request to execute a tool with specific arguments.

## 26. What Is LangGraph?

LangGraph is a framework for building stateful agent workflows using graph concepts like state, nodes, edges, conditional routing, persistence, and human-in-the-loop.

## 27. LangChain vs LangGraph?

LangChain provides LLM application components. LangGraph provides reliable workflow orchestration for stateful, multi-step agents.

## 28. What Is State in LangGraph?

State is shared data carried through the graph. Nodes read state and return updates.

## 29. What Is a Node?

A node is a function that performs one step in the graph.

## 30. What Is an Edge?

An edge connects nodes and decides the next step.

## 31. What Is Persistence?

Persistence saves graph state as checkpoints so workflows can resume, support memory, and enable debugging.

## 32. What Is Time Travel?

Time travel means inspecting or returning to previous graph checkpoints.

## 33. What Is Human in the Loop?

Human in the loop means the workflow pauses for human review, approval, or correction before continuing.

## 34. What Is Short-Term Memory?

Short-term memory stores context within one conversation or thread.

## 35. What Is Long-Term Memory?

Long-term memory stores useful information across conversations, such as user preferences or stable facts.

## 36. Memory vs RAG?

Memory remembers user or conversation information. RAG retrieves external knowledge from documents.

## 37. What Is MCP?

MCP is Model Context Protocol, a standard way for AI applications to connect to tools, resources, and prompts exposed by MCP servers.

## 38. MCP Host, Client, Server?

The host is the AI application. The client maintains a connection to one server. The server exposes tools, resources, and prompts.

## 39. MCP Tools vs Resources?

Tools are actions. Resources are readable context/data.

## 40. MCP Local vs Remote?

Local MCP servers often use STDIO. Remote MCP servers use HTTP-based transport.

## 41. What Is Ollama?

Ollama runs LLMs locally through a CLI and local API. It supports chat models and embedding models.

## 42. Why Use Ollama?

Use it for privacy, offline use, local experimentation, lower cost, and local RAG.

## 43. Cloud LLM vs Local LLM?

Cloud LLMs are usually stronger and easier to scale. Local LLMs offer privacy and control but depend on hardware and may be weaker.

## 44. What Is Evaluation in GenAI?

Evaluation measures whether the system works. For RAG, evaluate retrieval relevance and answer faithfulness. For agents, evaluate task success, tool correctness, latency, and safety.

## 45. What Is Hallucination?

Hallucination is when the model gives unsupported or false information confidently.

## 46. How Do You Reduce Hallucination?

Use RAG, citations, tool calls, structured output, validation, better prompts, and evaluation.

## 47. How Would You Build a PDF Chatbot?

Load PDFs, split text, create embeddings, store in vector DB, retrieve relevant chunks for each question, pass context to LLM, answer with citations, and evaluate retrieval/answer quality.

## 48. How Would You Build a YouTube Chatbot?

Load transcript, split by topic/timestamp, embed chunks, store vectors, retrieve relevant transcript chunks, answer with timestamp citations.

## 49. What Are Production Concerns?

Security, access control, prompt injection, hallucination, latency, cost, evaluation, observability, retries, fallbacks, and user feedback.

## 50. Final 60-Second Answer

```text
A modern GenAI application starts with an LLM, but production systems need more than a model. We design prompts, parse structured outputs, and use tools when the model must act. For private or changing knowledge, we use RAG: load documents, split them, embed them, store vectors, retrieve relevant chunks, and generate grounded answers with citations. LangChain helps compose these components. LangGraph helps build reliable stateful workflows with nodes, edges, persistence, streaming, memory, and human approval. MCP standardizes how AI apps connect to external tools and data. Ollama lets us run local models and embeddings. Finally, we evaluate the system for retrieval quality, faithfulness, latency, cost, and safety.
```
