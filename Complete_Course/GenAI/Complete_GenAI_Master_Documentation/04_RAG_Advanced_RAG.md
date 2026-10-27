# Retrieval-Augmented Generation (RAG) & Advanced RAG

This document explains the technical details of Naive RAG, advanced document ingestion/splitting, search mechanisms, re-ranking, advanced agentic architectures (CRAG, Self-RAG, GraphRAG), and evaluation frameworks.

---

## 1. What is RAG?

Retrieval-Augmented Generation (RAG) combines a retrieval model (which finds relevant passages in a vector store) with a generative model (an LLM which produces an answer based on those passages). 

It resolves the three biggest limitations of LLMs:
1.  **Hallucinations:** Limits answers to factual retrieved text.
2.  **Stale Knowledge:** Fresh documents can be added to the vector store instantly without re-training models.
3.  **Data Security:** Allows search across private company internal databases securely.

---

## 2. Ingestion & Advanced Chunking

### Advanced Chunking Strategies
*   **Recursive Character Chunking:** Splits text by paragraph, then sentences, then words, aiming to keep related ideas together within size constraints.
*   **Parent-Child Chunking:**
    *   **Child Chunks (e.g. 100 tokens):** Indexed and compared against the query vector for fine-grained semantic match.
    *   **Parent Chunks (e.g. 1000 tokens):** Stored as the actual text associated with the child chunks. When a child chunk is retrieved, the parent chunk is passed to the LLM. This preserves broad contextual flow.
*   **Semantic Chunking:** Calculates the embedding vector for each sentence. If the cosine distance between two consecutive sentences exceeds a set threshold, it creates a split. This splits documents at natural topic changes rather than arbitrary character counts.

---

## 3. Search & Retrieval Optimization

```text
Query ===> [ Query Expansion ] ===> Multi-Queries ===> [ Search (Keyword + Vector) ]
                                                                 │
User <=== [ Generation ] <=== [ Re-ranking ] <=== Reciprocal Rank Fusion (RRF)
```

### 1. Hybrid Search
Combines two complementary search algorithms:
*   **Dense Retrieval (Vector Search):** Translates query and documents into vectors. Finds matches based on semantic meaning (synonyms, conceptual overlap).
*   **Sparse Retrieval (BM25 Keyword Search):** Counts word frequencies. Excellent for finding exact product SKU codes, medical names, or unique IDs.

### 2. Reciprocal Rank Fusion (RRF)
Merges ranked lists from Dense and Sparse searches. The score for document $d$ is:

$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where $r_m(d)$ is the rank of document $d$ in search system $m$, and $k$ is a constant (usually 60) to prevent top ranks from dominating.

### 3. Query Expansion
Users often search using vague phrases. We use an LLM to generate:
*   **Synonyms / Alternative phrasings:** Searches multiple variations of the query.
*   **HyDE (Hypothetical Document Embeddings):** The LLM generates a fake "perfect" answer to the query first. We then embed this fake answer and use it to search the vector database. Searching a document-like vector matches real documents better than searching a short question vector.

---

## 4. Re-ranking: Bi-Encoders vs. Cross-Encoders

To achieve maximum accuracy at low cost, modern RAG pipelines use a two-stage retrieval process:

| Stage | Model Type | How it Works | Speed/Cost |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Retrieval)** | **Bi-Encoder** | Encodes documents and query independently into vectors. Computes similarity using simple dot-product. | Extremely fast, low cost. Matches top 50 candidates. |
| **Stage 2 (Re-ranking)** | **Cross-Encoder** | Feeds the query and document *together* into the network, calculating full attention between every query word and every document word. | Slow, high cost, but extremely accurate. Reranks the top 50 down to the top 5. |

---

## 5. Next-Gen RAG Frameworks

### Corrective RAG (CRAG)
CRAG adds a quality control step before the LLM reads retrieved documents.
1.  **Retrieve:** Fetch document chunks.
2.  **Evaluate:** A lightweight classification model determines if the retrieved context is relevant.
3.  **Decide:**
    *   *Correct:* Route straight to LLM.
    *   *Incorrect:* Discard context and trigger a Web Search API query to retrieve correct facts.
    *   *Ambiguous:* Combine the retrieved documents and web search results.

### Self-RAG
Self-RAG trains an LLM to generate special **reflection tokens** that critique its own behavior during generation:
*   `[Retrieve]`: Decide if retrieval is needed.
*   `[IsRel]`: Evaluate if retrieved source text is relevant.
*   `[IsSup]`: Check if generated statements are supported by sources (no hallucination).
*   `[IsUse]`: Check if output actually answers the question.

### GraphRAG
Instead of splitting documents into standard text blocks, GraphRAG:
1.  Uses an LLM to extract entities (e.g. people, places, dates) and their connections (e.g. "works at", "owns").
2.  Stores them in a **Knowledge Graph**.
3.  Allows the retriever to follow paths across the graph, making it excellent at answering high-level, global questions (e.g. *"What are the main topics discussed in these 100 documents?"*) that standard vector search misses.

---

## 6. RAG Evaluation Metrics (RAGAS)

We evaluate RAG using four core metrics:
1.  **Faithfulness:** Is the generated answer based *only* on the retrieved context? (Low score = model is hallucinating).
2.  **Answer Relevance:** Does the answer directly address the user's question? (Low score = model is rambling or off-topic).
3.  **Context Recall:** Did the retriever find all necessary information to answer the question?
4.  **Context Precision:** Are all retrieved chunks highly relevant, or is there a lot of useless noise?
