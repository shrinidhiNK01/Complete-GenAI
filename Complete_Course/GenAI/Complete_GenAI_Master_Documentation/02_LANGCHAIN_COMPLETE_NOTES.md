# LangChain Complete Notes

## 1. What LangChain Is

LangChain is a framework for building LLM applications. It helps connect models, prompts, parsers, retrievers, tools, and agents.

The important idea is composition. Instead of writing everything manually, you connect reusable components.

## 2. LangChain Core Building Blocks

### Models

Models are the LLMs or chat models you call.

Examples:

- OpenAI models
- Anthropic models
- Gemini models
- Ollama local models
- Hugging Face models

Model input is usually a list of messages. Model output is usually an AI message.

### Prompts

Prompts format instructions and variables.

Example:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a simple English teacher."),
    ("user", "Explain {topic} in 5 bullet points.")
])
```

### Output Parsers

Output parsers convert model text into useful format.

Examples:

- Plain string parser
- JSON parser
- Pydantic parser
- List parser

Use parsers when your application needs reliable data, not only text.

### Runnables

Runnable is the common interface for LangChain components.

Important methods:

- `invoke`: run once
- `batch`: run many inputs
- `stream`: stream output
- `ainvoke`: async run

LCEL, LangChain Expression Language, lets you compose components with `|`.

Example:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "RAG"})
```

### Documents

LangChain uses `Document` objects to store text plus metadata.

```python
Document(
    page_content="Refund policy says refunds take 5 days.",
    metadata={"source": "policy.pdf", "page": 2}
)
```

### Document Loaders

Loaders ingest data from files and apps.

Examples:

- PDF loader
- CSV loader
- Web loader
- Notion loader
- YouTube transcript loader
- Directory loader

Loader output is usually a list of `Document` objects.

### Text Splitters

Text splitters break large documents into chunks.

Why splitting is needed:

- LLM context is limited.
- Retrieval works better on focused chunks.
- Embeddings represent smaller meaning units better.

Common splitting strategies:

- Recursive character splitting
- Token splitting
- Markdown/header-aware splitting
- Code-aware splitting
- Semantic splitting

Chunk parameters:

- `chunk_size`: size of each chunk
- `chunk_overlap`: repeated text between chunks

Typical values:

- General docs: chunk size 500 to 1200 tokens
- Overlap: 50 to 200 tokens
- Code: split by functions/classes when possible

### Embeddings

Embedding models convert text to vectors.

Use embeddings for:

- Semantic search
- RAG
- Clustering
- Similarity comparison
- Recommendation

### Vector Stores

Vector stores save embeddings and allow similarity search.

Common options:

- FAISS: local, fast, simple
- Chroma: local or server, beginner friendly
- Pinecone: managed cloud
- Weaviate: vector DB with hybrid search
- Milvus: scalable vector DB
- Qdrant: vector DB
- pgvector: PostgreSQL extension

### Retrievers

Retrievers return documents for a query.

Types:

- Vector retriever
- Keyword retriever
- Hybrid retriever
- Multi-query retriever
- Parent document retriever
- Contextual compression retriever
- Ensemble retriever

### Tools

Tools are callable functions exposed to a model or agent.

Example:

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

Tool descriptions matter because the model uses them to decide when and how to call tools.

### Agents

Agents use a model and tools in a loop.

Modern LangChain agents are built on LangGraph. This gives agents durable execution, persistence, and human-in-the-loop support.

Basic agent loop:

```text
User input
Model decides whether to call tool
Tool executes
Result returns to model
Model decides next step
Final answer
```

## 3. Simple LangChain Flow

```text
User input -> prompt template -> model -> output parser -> final result
```

Code:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain topics in simple English."),
    ("user", "Explain {topic}.")
])

chain = prompt | model | StrOutputParser()

print(chain.invoke({"topic": "LangChain"}))
```

## 4. Prompt Engineering in LangChain

A good prompt usually contains:

- Role
- Task
- Context
- Rules
- Output format
- Examples

Example:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an interview coach.
Explain concepts in simple English.
Use short examples.
If the answer is uncertain, say what is missing.
"""),
    ("user", "Teach me {concept}.")
])
```

## 5. Structured Output

Structured output means the model returns data in a predictable shape.

Why it matters:

- Easier to parse
- Easier to validate
- Better for APIs
- Better for UI rendering
- Less fragile than regex parsing

Example with Pydantic:

```python
from pydantic import BaseModel, Field

class InterviewAnswer(BaseModel):
    topic: str = Field(description="Topic being answered")
    short_answer: str
    example: str
    common_mistake: str

structured_model = model.with_structured_output(InterviewAnswer)
result = structured_model.invoke("Explain RAG")
print(result.topic)
```

In agents, LangChain can use provider-native structured output when supported, or tool-calling based structured output otherwise.

## 6. Output Parsers

Use output parsers when:

- You need plain text
- You need JSON
- You need lists
- You need typed objects
- You need to post-process model output

Example:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | model | StrOutputParser()
```

## 7. Chains

A chain is a sequence of steps.

Old LangChain used many special chain classes. Modern LangChain prefers runnables and LCEL.

Example:

```python
chain = prompt | model | parser
```

You can also create parallel inputs:

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = {
    "context": retriever,
    "question": RunnablePassthrough()
} | prompt | model | parser
```

## 8. Runnables Deep Explanation

Runnable is like a standard plug socket. If a component is runnable, it can be connected with other runnables.

Important runnable features:

- Sync and async support
- Streaming
- Batching
- Retries
- Configuration
- Composition

Common runnable types:

- Prompt templates
- Models
- Output parsers
- Lambda functions
- Retrievers
- Tools

## 9. Document Loaders

A document loader converts raw source into documents.

Example sources:

- PDF
- TXT
- Markdown
- HTML
- CSV
- YouTube transcript
- Website
- Database

Good metadata is very important. Always store source information:

```python
metadata={
    "source": "employee_handbook.pdf",
    "page": 12,
    "section": "Leave Policy"
}
```

Without metadata, you cannot cite sources properly.

## 10. Text Splitters

Bad chunking gives bad RAG.

Chunking goals:

- Each chunk should contain one meaningful idea.
- Chunks should not be too big.
- Chunks should not lose important context.
- Overlap should preserve continuity.

Recursive splitter tries larger separators first, then smaller ones.

Common separator order:

```text
paragraph -> sentence -> word -> character
```

## 11. Vector Stores and Retrievers

Vector store stores embeddings. Retriever wraps search behavior.

Similarity search:

```text
Find chunks closest to the query embedding.
```

MMR, Maximal Marginal Relevance:

```text
Return relevant chunks but avoid duplicates.
```

Hybrid search:

```text
Combine keyword search and vector search.
```

Use hybrid search when exact terms matter, such as product IDs, legal clauses, error codes, or names.

## 12. Tools in LangChain

Tool design rules:

- Tool name should be clear.
- Description should say when to use it.
- Arguments should be typed.
- Tool should do one thing.
- Tool should return concise useful result.
- Dangerous tools need human approval.

Bad tool:

```python
@tool
def do_stuff(x: str): ...
```

Good tool:

```python
@tool
def search_policy(query: str) -> str:
    """Search company policy documents for a policy-related question."""
```

## 13. Agents in LangChain

Use agents when the model must decide what to do.

Do not use agents when the steps are fixed. Use a chain or graph instead.

Good agent use cases:

- Research assistant
- SQL assistant
- Customer support assistant
- Data analysis assistant
- Tool-using chatbot

Bad agent use cases:

- Simple summarization
- Simple classification
- One-step RAG
- Fixed extraction task

## 14. LangSmith and Observability

Observability means seeing what happened inside the LLM app.

Track:

- Prompt
- Model input/output
- Retrieved chunks
- Tool calls
- Latency
- Token usage
- Cost
- Errors
- User feedback

LangSmith is commonly used with LangChain/LangGraph for tracing and evaluation.

## 15. LangChain Interview Summary

Say this:

```text
LangChain helps build LLM apps by composing models, prompts, parsers, retrievers, tools, and agents. Its modern design uses the Runnable interface and LCEL so components can be invoked, streamed, batched, and composed. For private knowledge, LangChain supports document loaders, text splitters, embeddings, vector stores, and retrievers. For action-taking systems, LangChain agents use tools and are built on LangGraph for reliable orchestration.
```
