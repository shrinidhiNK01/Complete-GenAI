# Big Picture Beginner Notes

## 1. What Is Generative AI?

Generative AI is AI that creates new content. It can generate text, code, images, audio, video, summaries, answers, plans, and structured data.

Traditional software follows fixed rules written by humans. Generative AI uses learned patterns from large amounts of data. You give it an input, usually called a prompt, and it generates an output.

Examples:

- Ask a chatbot to explain Python.
- Ask an AI assistant to summarize a PDF.
- Ask a coding model to write SQL.
- Ask an AI agent to search documents, call tools, and produce a report.

## 2. What Is an LLM?

LLM means Large Language Model. It is a model trained on text and code to predict the next token.

A token is a small piece of text. It can be a word, part of a word, a punctuation mark, or a symbol. The model reads tokens and predicts the next likely tokens.

Important point: an LLM does not truly "know" like a human. It generates likely text based on patterns. It can still reason surprisingly well, but it can also hallucinate.

## 3. Core LLM Terms

### Prompt

The input instruction you give to the model.

Example:

```text
Explain RAG in simple English with one example.
```

### Completion / Response

The generated answer from the model.

### Context Window

The maximum amount of text the model can consider at once. If the context window is 128k tokens, the model can read up to around that many tokens in the prompt plus conversation plus retrieved context plus output.

### Temperature

Controls randomness.

- Low temperature, such as `0` to `0.3`: more deterministic, better for facts and code.
- Medium temperature, such as `0.5` to `0.8`: more creative.
- High temperature: more random, often risky for production.

### Top-p

Another sampling control. It limits token choices to a probability mass. In most applications, set either temperature or top-p, not both aggressively.

### System Message

High-priority instruction that defines behavior.

Example:

```text
You are a strict SQL assistant. Return only valid PostgreSQL.
```

### User Message

The user's actual query.

### Assistant Message

Previous model replies.

### Tool Message

The result of a tool call returned back to the model.

## 4. Why LLMs Hallucinate

Hallucination means the model gives an answer that sounds confident but is wrong.

Reasons:

- The model is trained to generate likely text, not guaranteed truth.
- Its training knowledge may be outdated.
- The prompt may not provide enough facts.
- The retrieved context may be wrong or incomplete.
- The model may confuse similar concepts.

How to reduce hallucination:

- Use RAG to provide trusted context.
- Ask the model to cite sources.
- Use structured output and validation.
- Add retrieval evaluation.
- Use tools for live data.
- Keep prompts clear and specific.
- Do not ask the model to guess when facts are missing.

## 5. What Is LangChain?

LangChain is a framework for building applications with LLMs. It gives reusable components for:

- Models
- Prompts
- Output parsing
- Document loading
- Text splitting
- Embeddings
- Vector stores
- Retrievers
- RAG pipelines
- Tools
- Agents
- Structured output
- Streaming
- Observability integrations

Simple mental model:

```text
Prompt + Model + Parser = basic LLM app
Documents + Embeddings + Vector Store + Retriever + Model = RAG app
Model + Tools + Loop = agent app
```

## 6. What Is LangGraph?

LangGraph is a low-level orchestration framework for building stateful, reliable agents and workflows.

LangChain is good for connecting LLM components. LangGraph is good when you need control:

- Multi-step workflows
- Branching logic
- Loops
- Human approval
- Persistence
- Resume after failure
- Time travel debugging
- Long-running agents
- Stateful conversations
- Subgraphs
- Streaming

Simple mental model:

```text
State + Nodes + Edges = Graph workflow
```

Each node performs work. Edges decide where to go next. State carries data through the graph.

## 7. What Is RAG?

RAG means Retrieval-Augmented Generation.

Problem: an LLM cannot know everything, and it cannot read your entire company's documents every time.

Solution:

1. Store your documents in a searchable knowledge base.
2. When the user asks a question, retrieve relevant chunks.
3. Give those chunks to the LLM.
4. The LLM answers using that context.

Simple RAG pipeline:

```text
Documents -> Split into chunks -> Embed chunks -> Store vectors
User question -> Embed question -> Retrieve similar chunks -> Prompt LLM -> Answer
```

## 8. What Are Embeddings?

Embeddings are numeric vectors representing meaning.

Example:

- "doctor" and "hospital" will have close vectors.
- "pizza" and "database migration" will be far apart.

Embeddings allow semantic search. Semantic search means searching by meaning, not exact words.

## 9. What Is a Vector Store?

A vector store is a database for embeddings.

It stores:

- Text chunk
- Embedding vector
- Metadata

It supports similarity search. Popular vector stores include FAISS, Chroma, Pinecone, Weaviate, Milvus, Qdrant, and pgvector.

## 10. What Is a Retriever?

A retriever takes a query and returns relevant documents.

In LangChain, a retriever usually returns `Document` objects. A document has:

- `page_content`: the text
- `metadata`: source, page number, URL, title, etc.

## 11. What Are Tools?

A tool is a function the model can call.

Examples:

- Search the web
- Query a database
- Send email
- Run Python
- Retrieve documents
- Calculate math
- Create a ticket
- Call an API

Tools make LLMs useful because the model can act on the outside world.

## 12. What Is Tool Calling?

Tool calling means the model returns a structured request saying:

```text
Call tool `get_weather` with city = "Bengaluru".
```

Then your application executes the tool and sends the result back to the model.

Tool calling is better than asking the model to write fake JSON in plain text because the API/framework can validate tool names and arguments.

## 13. What Is an Agent?

An agent is an LLM system that can decide steps.

Basic chain:

```text
Input -> fixed step 1 -> fixed step 2 -> output
```

Agent:

```text
Input -> model decides -> tool call -> observe result -> model decides again -> final answer
```

An agent usually has:

- Model
- Tools
- Prompt
- State
- Loop
- Stop condition

## 14. Agentic AI vs Generative AI

Generative AI creates content.

Agentic AI makes decisions and takes actions over multiple steps.

Example:

- Generative AI: "Write an email."
- Agentic AI: "Find the customer complaint, check order status, draft a reply, create refund ticket, ask manager approval if refund is above limit."

## 15. What Is Memory?

Memory lets an AI system remember useful information.

There are two main types.

### Short-Term Memory

Memory within a conversation or thread.

Example:

User: "My name is Nidhi."

Later in same thread:

User: "What is my name?"

The assistant remembers because the conversation state is still available.

### Long-Term Memory

Memory across conversations.

Example:

The assistant remembers:

- User prefers simple English.
- User is preparing for GenAI interviews.
- User mostly codes in Python.

Long-term memory should be stored carefully, with privacy and user control.

## 16. What Is Persistence?

Persistence means saving state so the app can resume.

Without persistence:

- Server crashes, conversation is lost.
- Human approval workflow cannot pause and resume.
- Agent cannot continue a long task after interruption.

With persistence:

- State is checkpointed.
- Threads can resume.
- You can inspect previous graph steps.
- You can do time travel debugging.

## 17. What Is Human in the Loop?

Human in the loop means the AI pauses and waits for human approval or correction.

Use it when:

- Money is involved.
- Legal or medical risk exists.
- Data deletion is possible.
- The agent wants to send an external message.
- The answer needs review.

Example:

```text
Agent drafts refund -> pauses -> manager approves -> agent executes refund
```

## 18. What Is Streaming?

Streaming means showing model output token by token or event by event instead of waiting for the full answer.

Benefits:

- Better user experience.
- User sees progress.
- Useful for long answers.
- Can stream intermediate agent events, not only final text.

## 19. What Is MCP?

MCP means Model Context Protocol.

It is a standard way for AI apps to connect to external tools and data sources.

Without MCP, every AI app needs custom integration for every tool.

With MCP:

```text
AI app -> MCP client -> MCP server -> tools/resources/prompts
```

MCP server can expose:

- Tools: actions the model can call
- Resources: data the model can read
- Prompts: reusable prompt templates

## 20. What Is Ollama?

Ollama lets you run open models locally on your machine.

It gives:

- CLI
- Local REST API
- Model library
- Embeddings
- Python and JavaScript libraries

Common use cases:

- Local chatbot
- Local RAG
- Privacy-sensitive document assistant
- Testing without cloud API costs

## 21. Full System Architecture Example

Example: company policy chatbot.

```text
User asks question
       |
Frontend app
       |
Backend API
       |
LangGraph workflow
       |
Retriever node gets policy chunks
       |
LLM node writes answer
       |
Validation node checks citation coverage
       |
Final answer with sources
```

Data ingestion:

```text
PDFs -> loader -> splitter -> embeddings -> vector store
```

Runtime:

```text
Question -> retriever -> context -> model -> answer
```

## 22. What You Must Say in Interviews

If asked "How do you build a GenAI app?", answer like this:

```text
First I define the task and success metric. Then I choose the model, design the prompt, and decide whether I need tools or RAG. If the app needs private or changing knowledge, I build a retrieval pipeline with loaders, splitters, embeddings, vector store, and retriever. If it needs multi-step decisions, I use an agent or LangGraph. I add memory only when needed. Finally, I evaluate for quality, hallucination, latency, cost, and safety before production.
```

## 23. Common Beginner Mistakes

- Thinking the LLM database contains your private documents.
- Using RAG without evaluating retrieved chunks.
- Using huge chunks that do not fit the prompt well.
- Forgetting metadata and citations.
- Letting agents call dangerous tools without approval.
- Confusing memory with RAG.
- Confusing LangChain and LangGraph.
- Building an agent when a simple chain is enough.
- Not validating structured outputs.
- Ignoring latency and cost.

## 24. Final Beginner Summary

You need only this core mental map:

```text
LLM = generates text
Prompt = instruction
Parser = converts output
Embedding = meaning vector
Vector store = searchable embedding DB
Retriever = finds relevant chunks
RAG = retrieve then generate
Tool = external function
Agent = model decides actions
LangChain = LLM application components
LangGraph = reliable agent workflow orchestration
Memory = remembers context
MCP = standard tool/data connection protocol
Ollama = run models locally
Evaluation = proves quality
```
