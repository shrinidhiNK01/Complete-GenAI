# Code Cookbook

This file keeps the most important code patterns in one place.

## 1. Install Packages

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters langgraph
pip install chromadb faiss-cpu pypdf python-dotenv streamlit
pip install langchain-ollama
```

## 2. Basic LangChain Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Explain in simple English."),
    ("user", "Explain {topic}.")
])

chain = prompt | model | StrOutputParser()

print(chain.invoke({"topic": "RAG"}))
```

## 3. Structured Output With Pydantic

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class ConceptAnswer(BaseModel):
    concept: str = Field(description="The concept name")
    simple_definition: str
    example: str
    interview_line: str

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_model = model.with_structured_output(ConceptAnswer)

result = structured_model.invoke("Explain LangGraph")
print(result.simple_definition)
```

## 4. Create a Tool

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

## 5. Create a Simple Agent

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return fake weather for a city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Bengaluru?"}]
})

print(result["messages"][-1].content)
```

## 6. Load a PDF

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy.pdf")
docs = loader.load()

print(len(docs))
print(docs[0].page_content[:500])
print(docs[0].metadata)
```

## 7. Split Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)
print(len(chunks))
```

## 8. Create a Vector Store With Chroma

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

## 9. Basic RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata}\n{doc.page_content}"
        for doc in docs
    )

prompt = ChatPromptTemplate.from_template("""
Use only the context to answer.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}
""")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("What is the sick leave policy?"))
```

## 10. RAG With Sources

```python
def answer_with_sources(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = (prompt | model | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    sources = [doc.metadata for doc in docs]
    return {"answer": answer, "sources": sources}
```

## 11. LangGraph Basic Workflow

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State):
    return {"answer": f"You asked: {state['question']}"}

builder = StateGraph(State)
builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()

print(graph.invoke({"question": "What is LangGraph?"}))
```

## 12. LangGraph Conditional Routing

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    route: str
    answer: str

def classify(state: State):
    if "policy" in state["question"].lower():
        return {"route": "rag"}
    return {"route": "direct"}

def route_next(state: State) -> Literal["rag", "direct"]:
    return state["route"]

def rag_node(state: State):
    return {"answer": "RAG answer would be generated here."}

def direct_node(state: State):
    return {"answer": "Direct LLM answer would be generated here."}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("rag", rag_node)
builder.add_node("direct", direct_node)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_next)
builder.add_edge("rag", END)
builder.add_edge("direct", END)

graph = builder.compile()
```

## 13. LangGraph Persistence Example

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "interview-study-thread"}}

graph.invoke({"question": "Explain RAG"}, config=config)
graph.invoke({"question": "Now explain embeddings"}, config=config)
```

Use database-backed checkpointers for production.

## 14. Ollama With LangChain

```python
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2", temperature=0)
response = model.invoke("Explain MCP in simple English.")
print(response.content)
```

## 15. Ollama Embeddings

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="embeddinggemma")
query_vector = embeddings.embed_query("What is RAG?")
print(len(query_vector))
```

## 16. Local RAG With Ollama and Chroma

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="embeddinggemma")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./local_chroma"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

## 17. Streamlit Chatbot Skeleton

```python
import streamlit as st

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    answer = rag_chain.invoke(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
```

## 18. Simple MCP Server Conceptual Example

Exact SDK code can change, but the concept is always:

```python
# Pseudocode
server = MCPServer("company-tools")

@server.tool()
def search_policy(query: str) -> str:
    """Search company policy documents."""
    return policy_search(query)

@server.resource("schema://hr")
def hr_schema() -> str:
    return "employees(id, name, department, leave_balance)"

server.run_stdio()
```

## 19. RAG Evaluation Skeleton

```python
questions = [
    {
        "question": "How many sick leaves are allowed?",
        "expected_source": "leave_policy.pdf"
    }
]

for item in questions:
    docs = retriever.invoke(item["question"])
    sources = [doc.metadata.get("source") for doc in docs]
    hit = item["expected_source"] in sources
    print(item["question"], "PASS" if hit else "FAIL", sources)
```

## 20. Safe Tool Approval Pattern

```python
def send_email(to: str, subject: str, body: str, approved: bool = False):
    if not approved:
        return {
            "status": "approval_required",
            "preview": {"to": to, "subject": subject, "body": body}
        }
    # Actually send email here
    return {"status": "sent"}
```
