# RAG Complete Guide

## 1. What Is RAG?

RAG means Retrieval-Augmented Generation.

It combines two things:

1. Retrieval: find relevant information from external sources.
2. Generation: use an LLM to answer using that information.

RAG is used when the LLM needs knowledge that is:

- Private
- Recent
- Domain-specific
- Too large to fit in the prompt
- Required with citations

## 2. Why RAG Is Needed

LLMs have limitations:

- Training knowledge becomes outdated.
- They cannot access your private files unless you give them.
- They have context limits.
- They can hallucinate.

RAG reduces these problems by grounding the answer in retrieved documents.

## 3. Basic RAG Architecture

Indexing phase:

```text
Documents
  -> Load
  -> Split
  -> Embed
  -> Store in vector DB
```

Query phase:

```text
User question
  -> Embed query
  -> Retrieve top chunks
  -> Put chunks into prompt
  -> LLM generates answer
  -> Return answer with sources
```

## 4. Indexing Step by Step

### Step 1: Load Documents

Use loaders for PDFs, text files, websites, CSVs, YouTube transcripts, databases, or APIs.

Always store metadata:

- Source file
- Page number
- Section
- URL
- Created date
- Access permissions

### Step 2: Clean Documents

Cleaning examples:

- Remove headers and footers.
- Remove repeated navigation text.
- Normalize whitespace.
- Fix OCR errors.
- Remove irrelevant pages.
- Preserve tables carefully.

### Step 3: Split Text

Chunking is very important.

Bad chunking example:

```text
Chunk contains half of policy A and half of policy B.
```

Good chunking example:

```text
Chunk contains one complete leave policy section with heading.
```

Common chunking methods:

- Recursive character splitter
- Token splitter
- Markdown heading splitter
- HTML section splitter
- Code splitter
- Semantic splitter
- Parent-child splitter

### Step 4: Embed Chunks

Embedding converts each chunk into a numeric vector.

Choose embedding model based on:

- Language
- Domain
- Cost
- Speed
- Dimension
- Retrieval accuracy

### Step 5: Store in Vector Store

Store:

- Text
- Embedding
- Metadata
- Document ID

Use metadata filters for:

- User access control
- Department
- Date range
- Product
- Document type

## 5. Query Step by Step

### Step 1: Receive User Question

Example:

```text
Can I take sick leave during probation?
```

### Step 2: Rewrite or Clean Query

Sometimes the user question is vague.

Query rewriting can convert:

```text
What about probation?
```

into:

```text
What is the sick leave policy for employees during probation?
```

### Step 3: Retrieve Documents

Retrieve top `k` chunks.

Common values:

- `k=3` for short answers
- `k=5` to `8` for detailed answers
- More only if reranking or compression is used

### Step 4: Rerank

Reranking means using a stronger model to reorder retrieved chunks by relevance.

Useful when vector search returns many possible chunks.

### Step 5: Build Prompt

Prompt should include:

- Role
- Context
- Question
- Rules
- Citation requirement
- What to do if context is insufficient

Example:

```text
Answer using only the context.
If the context does not contain the answer, say "I do not know from the provided documents."
Include source citations.
```

### Step 6: Generate Answer

The model writes the final answer grounded in context.

### Step 7: Validate

Check:

- Did it cite sources?
- Did it answer from context?
- Did it hallucinate?
- Did retrieval find correct chunks?

## 6. Types of RAG

### 2-Step RAG

Fixed process:

```text
retrieve -> answer
```

Best for:

- FAQ
- Policy bot
- Simple document Q&A

### Agentic RAG

The model decides when and how to retrieve.

Example:

```text
Question -> agent decides search docs -> searches -> sees weak results -> rewrites query -> searches again -> answers
```

Best for:

- Complex research
- Multi-hop questions
- Tool-using assistants

### Hybrid RAG

Combines vector search and keyword search.

Best for:

- Legal docs
- Error codes
- Product names
- Acronyms
- Exact IDs

### Graph RAG

Uses graph relationships between entities.

Best for:

- Knowledge graphs
- Relationship-heavy data
- Multi-hop reasoning over entities

### Corrective RAG, CRAG

Corrective RAG checks whether retrieved documents are good enough.

Flow:

```text
Retrieve docs
Grade docs
If good -> answer
If bad -> rewrite query or search web/other source
Then answer
```

Main idea: do not blindly trust first retrieval.

### Self-RAG

Self-RAG asks the model to critique retrieval and answer quality.

It may decide:

- Need retrieval
- Retrieved docs are relevant or not
- Answer is supported or not
- Need to revise

Main idea: model participates in judging its own grounding.

## 7. Retrieval Techniques

### Similarity Search

Find nearest vectors to the query.

Pros:

- Simple
- Fast
- Good semantic matching

Cons:

- May miss exact keywords
- Can return duplicates

### MMR Search

Maximal Marginal Relevance balances relevance and diversity.

Use when top results are too similar.

### Keyword Search

Search exact words using BM25 or similar methods.

Use when exact terms matter.

### Hybrid Search

Combines vector and keyword search.

This is often stronger than pure vector search in production.

### Multi-Query Retrieval

Generate multiple versions of the user question, search each, and merge results.

Example:

```text
Original: "Can probation employees take leave?"
Query 1: "probation sick leave policy"
Query 2: "leave eligibility during probation"
Query 3: "new employee leave rules"
```

### Parent Document Retrieval

Store small chunks for search but return larger parent sections for context.

Why:

- Small chunks retrieve accurately.
- Larger parent text gives full context.

### Contextual Compression

Retrieve documents, then compress them to only relevant lines.

Useful when chunks are long.

## 8. Chunking Best Practices

Good chunking depends on document type.

### PDFs

- Preserve page numbers.
- Watch out for tables.
- Remove headers/footers.
- Use OCR only when needed.

### Markdown

- Split by headings.
- Keep heading path in metadata.

### Code

- Split by functions/classes.
- Keep imports and surrounding context when needed.

### Legal/Policy Documents

- Split by clauses.
- Keep clause numbers.
- Preserve exact wording.

### YouTube Transcripts

- Split by topic and timestamp.
- Store video title and timestamp in metadata.

## 9. RAG Prompt Template

```text
You are a helpful assistant.
Use only the provided context to answer the question.
If the answer is not in the context, say you do not know.
Do not invent facts.
Give a short answer first, then details.
Cite sources using the metadata.

Context:
{context}

Question:
{question}
```

## 10. RAG Evaluation

You must evaluate both retrieval and generation.

### Retrieval Metrics

- Precision: are retrieved chunks relevant?
- Recall: did we retrieve all needed chunks?
- MRR: is the first useful result near the top?
- nDCG: ranking quality

### Generation Metrics

- Faithfulness: is answer supported by context?
- Answer relevance: does it answer the question?
- Correctness: is it factually right?
- Citation accuracy: do citations match claims?
- Completeness: did it miss key points?

### Human Evaluation

Ask reviewers:

- Is the answer correct?
- Is it grounded in sources?
- Is it easy to understand?
- Are citations useful?

## 11. Common RAG Failures

### Retrieval Miss

Correct answer exists but retriever does not find it.

Fix:

- Improve chunking.
- Use hybrid search.
- Add query rewriting.
- Improve embeddings.
- Add metadata filters.

### Bad Context

Retriever returns irrelevant chunks.

Fix:

- Rerank.
- Use better metadata.
- Reduce noisy documents.
- Tune top `k`.

### Lost in the Middle

Important context is included but model ignores it.

Fix:

- Keep prompt shorter.
- Put key context near beginning or end.
- Use reranking.
- Compress context.

### Hallucinated Answer

Model invents details.

Fix:

- Stronger prompt.
- Require citations.
- Add faithfulness check.
- Use answer validation.

### Stale Index

Documents changed but vector DB is old.

Fix:

- Build re-indexing pipeline.
- Track document versions.
- Delete old embeddings.

## 12. Production RAG Checklist

- Document permissions are respected.
- Metadata is complete.
- Index updates are automated.
- Retrieval quality is measured.
- Answers include citations.
- Hallucination checks exist.
- Latency is acceptable.
- Costs are monitored.
- User feedback is collected.
- Logs do not expose secrets.
- Evaluation dataset exists.

## 13. RAG Interview Answer

Use this answer:

```text
RAG is Retrieval-Augmented Generation. It solves the problem that LLMs do not know private, recent, or very large knowledge bases. In RAG, we load documents, split them into chunks, convert chunks into embeddings, and store them in a vector database. At query time, we retrieve relevant chunks, put them into the prompt, and ask the LLM to answer using that context. A good RAG system needs good chunking, metadata, retrieval, reranking, citations, and evaluation for faithfulness and relevance.
```
