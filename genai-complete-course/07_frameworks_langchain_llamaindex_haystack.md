# 07. Frameworks: LangChain, LlamaIndex, Haystack, and Vector Databases

## Why Frameworks Exist

You can build LLM apps from scratch, but frameworks save time by providing:

- Document loaders.
- Text splitters.
- Embedding wrappers.
- Vector store integrations.
- Chains.
- Agents.
- Memory.
- Observability.
- Evaluation helpers.

Frameworks are tools, not magic. Learn the core ideas first.

## LangChain

LangChain is useful for building LLM workflows with many components.

Core ideas:

- Models.
- Prompts.
- Output parsers.
- Retrievers.
- Tools.
- Agents.
- Runnables.
- LCEL.

## LCEL

LCEL means LangChain Expression Language.

It lets you connect components like a pipeline.

Conceptual example:

```python
chain = prompt | model | parser
```

Meaning:

- `prompt` creates the final prompt.
- `model` generates output.
- `parser` converts output into usable format.

## LangServe

LangServe exposes LangChain chains as APIs.

Use when:

- You built a chain.
- You want an API endpoint quickly.
- You need playground-style testing.

## LangSmith

LangSmith helps with:

- Tracing.
- Debugging.
- Dataset evaluation.
- Prompt versions.
- Run history.

## LlamaIndex

LlamaIndex is strong for data ingestion and indexing.

Useful concepts:

- Documents.
- Nodes.
- Indexes.
- Query engines.
- Retriever engines.
- Response synthesizers.
- Metadata filters.

LlamaIndex is especially nice when your problem is "connect my private documents to an LLM."

## LangChain vs LlamaIndex

| Area | LangChain | LlamaIndex |
| --- | --- | --- |
| General app workflows | Strong | Good |
| Data indexing | Good | Strong |
| Agents | Strong | Good |
| RAG ingestion | Good | Strong |
| Ecosystem integrations | Very broad | Broad |
| Learning curve | Medium/high | Medium |

Use LangChain when workflow orchestration is central. Use LlamaIndex when data indexing and retrieval are central. Many teams use both.

## Haystack

Haystack by deepset is a framework for production NLP and LLM pipelines.

Good for:

- Search pipelines.
- RAG.
- Evaluation.
- Modular components.
- Enterprise-style architecture.

## Vector Databases

Vector DBs store embeddings and search by similarity.

Options:

- FAISS: local, fast, library-based.
- Chroma: simple local/dev vector DB.
- Pinecone: managed cloud vector DB.
- Weaviate: vector DB with schema and hybrid search.
- Qdrant: strong open-source vector DB.
- Milvus: scalable open-source vector DB.
- LanceDB: good for multimodal and local analytics.
- MongoDB Atlas Vector Search: vector search inside MongoDB.
- AstraDB: Cassandra-based managed DB with vector search.

## FAISS

Use FAISS for:

- Local prototypes.
- Fast similarity search.
- No external service.

Limitations:

- You must manage persistence, metadata, and scaling carefully.

## Pinecone

Use Pinecone for:

- Managed vector DB.
- Production scaling.
- Hosted operations.

Tradeoff:

- Service cost.
- External dependency.

## Weaviate

Use Weaviate for:

- Hybrid search.
- Schema-based objects.
- Managed or self-hosted deployments.

## Qdrant

Use Qdrant for:

- Open-source vector search.
- Filtering.
- Production deployments.
- Self-hosting or managed cloud.

## LanceDB

Use LanceDB for:

- Local-first vector search.
- Multimodal datasets.
- Fast development.

## Neo4j and Knowledge Graphs

Neo4j stores relationships.

Use it when:

- Relationship traversal matters.
- You need entity graphs.
- Questions require multi-hop reasoning.

Example:

```cypher
MATCH (c:Customer)-[:PLACED]->(o:Order)-[:CONTAINS]->(p:Product)
WHERE c.name = "Alice"
RETURN p.name
```

Line-by-line:

- `MATCH` finds graph patterns.
- `(c:Customer)` finds customer nodes.
- `[:PLACED]` finds relationships.
- `WHERE` filters by customer name.
- `RETURN` returns product names.

## Framework Selection Rule

Choose the smallest framework that solves the problem.

Examples:

- Simple API call: no framework.
- Basic document QA: LlamaIndex or small custom RAG.
- Complex chains and agents: LangChain/LangGraph.
- Enterprise search pipeline: Haystack.
- Graph relationships: Neo4j plus RAG.

