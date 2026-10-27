# LangChain Complete Master Documentation

This document explains the core components of LangChain, the LangChain Expression Language (LCEL), document processing pipelines, memory management, and structured tool calling.

---

## 1. What is LangChain?

LangChain is an open-source framework designed to simplify the creation of applications powered by Large Language Models. It provides standard interfaces for linking different models, databases, prompt templates, and external tools together in a chain.

---

## 2. Models & ChatModels

LangChain distinguishes between two types of model interfaces:
*   **LLMs:** Take a raw string as input and return a string. (Legacy interface, e.g. text completion).
*   **ChatModels:** Take a sequence of message objects as input and return a message object. This is the standard interface for modern models.

### Message Types
*   `HumanMessage`: Represents user input.
*   `AIMessage`: Represents model output.
*   `SystemMessage`: Instructs the model how to behave.
*   `ToolMessage`: Contains the output returned by an executed tool.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage(content="You are a sarcastic assistant."),
    HumanMessage(content="What is 2 + 2?")
]

response = model.invoke(messages)
print(response.content)
```

---

## 3. LangChain Expression Language (LCEL)

LCEL is a declarative way to compose chains together. It uses the pipe operator (`|`), similar to Unix pipe operations.

### Concept: `Prompt | Model | OutputParser`
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Composing the chain
chain = prompt | model | parser

# Running the chain
response = chain.invoke({"topic": "programming"})
print(response)
```

### Why use LCEL?
*   **Streaming Support:** The chain inherits streaming capabilities automatically.
*   **Async Support:** Can run asynchronously using `.ainvoke()`.
*   **Parallel Execution:** Multiple components can run in parallel using `RunnableParallel`.
*   **Automatic Tracing:** Integrates with LangSmith out-of-the-box for logging and auditing.

---

## 4. Document Processing (RAG Foundations)

To feed private files into an LLM, LangChain provides a standard data preparation pipeline:

```text
Loaders (Read Files) ===> Splitters (Cut into Chunks) ===> VectorStore (Store & Search)
```

### 1. Document Loaders
Convert files into a standard `Document` object containing text content and metadata:
*   `PyPDFLoader`: Parses PDF pages.
*   `CSVLoader`: Reads rows of comma-separated files.
*   `DirectoryLoader`: Scans entire folders.

### 2. Text Splitters
Since models have limit windows, we split documents into smaller chunks. We use `RecursiveCharacterTextSplitter`, which splits by a list of characters (e.g. paragraphs, then sentences, then words) recursively to keep related sentences together.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max characters per chunk
    chunk_overlap=50,     # Overlap between consecutive chunks
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
```

### 3. VectorStores & Retrievers
*   **VectorStore:** Databases that store chunk texts alongside their embedding vectors (e.g. FAISS, ChromaDB, Pinecone).
*   **Retriever:** An interface that takes a query string and returns a list of relevant `Document` objects.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Embed and store chunks in-memory using FAISS
db = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Convert VectorStore to a Retriever interface
retriever = db.as_retriever(search_kwargs={"k": 3})
```

---

## 5. Tool Calling & Agents

### Declaring Tools
You can convert any Python function into a tool using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    # Custom API lookup logic here...
    return f"The weather in {location} is sunny, 22°C."
```

### Binding Tools to Models
Modern LLMs support native tool-calling. You bind your tools to the ChatModel so it knows which options are available:

```python
model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What is the weather in New York?")

# If the model decides it needs the tool, it returns a tool call proposal:
print(response.tool_calls)
# Output: [{'name': 'get_weather', 'args': {'location': 'New York'}, 'id': '...'}]
```

### LangChain Agents
An agent orchestrates the loop of calling tools and returning results automatically using standard templates (like `create_openai_tools_agent` or `create_react_agent`).
*   *Note:* For advanced multi-agent and cyclic tasks, LangGraph is preferred over standard LangChain Agents.
