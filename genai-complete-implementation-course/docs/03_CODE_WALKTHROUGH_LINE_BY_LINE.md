# 03. Code Walkthrough, File by File

This chapter explains the actual implementation project. Open the project at:

```text
projects/advanced_rag_llmops
```

## `app/core/config.py`

Purpose: store all tunable settings in one place.

Important code:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

Explanation:

- `__file__` is the path of `config.py`.
- `resolve()` converts it into an absolute path.
- `parents[2]` goes up from `app/core/config.py` to the project root.
- This avoids hardcoding machine-specific paths.

```python
@dataclass
class Settings:
```

Explanation:

- `@dataclass` creates a simple class for settings.
- You get an initializer automatically.
- Settings stay typed and readable.

```python
docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
```

This is where the source documents live.

```python
index_path: Path = PROJECT_ROOT / "storage" / "rag_index.json"
```

This is where the built index is saved.

```python
chunk_size_words: int = 120
chunk_overlap_words: int = 30
```

These control chunk size and overlap.

```python
def load_settings() -> Settings:
```

This function reads environment variables and returns a `Settings` object. In production, this is how Docker, Kubernetes, ECS, or Cloud Run changes behavior without changing code.

## `app/ingestion/models.py`

Purpose: define the data shapes.

```python
@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
```

Line-by-line:

- `Document` represents one source file or logical document.
- `doc_id` is a stable identifier.
- `text` is the extracted text.
- `metadata` stores extra information such as file name and source path.
- `default_factory=dict` prevents shared mutable defaults.

```python
@dataclass
class Chunk:
```

A `Chunk` is a smaller part of a document. RAG retrieves chunks, not usually entire documents.

Fields:

- `chunk_id`: unique chunk identifier.
- `doc_id`: parent document.
- `text`: chunk text.
- `start_word`: first word index.
- `end_word`: last word index.
- `metadata`: copied from the source document.

## `app/ingestion/loaders.py`

Purpose: read files into `Document` objects.

```python
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json"}
```

Only these file types are loaded by default. You can add PDF later with a PDF parser.

```python
def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")
```

This reads text safely. `errors="ignore"` avoids crashing on unusual characters.

```python
def load_csv_file(path: Path) -> str:
```

CSV rows are converted into readable sentences. This lets the same retrieval logic work on table-like data.

```python
def load_document(path: Path) -> Document:
```

This function chooses the correct loader based on file suffix, then creates a `Document`.

## `app/ingestion/chunking.py`

Purpose: clean and split documents.

```python
def clean_text(text: str) -> str:
```

It normalizes newlines and spaces.

```python
def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)
```

This splits text into word-like pieces. Production systems often use token-based splitters; this project uses words so the logic is easy to inspect.

```python
while start < len(words):
```

The loop creates chunk after chunk.

```python
end = min(start + chunk_size_words, len(words))
```

This chooses where the chunk ends without going past the document.

```python
start = end - overlap_words
```

This creates overlap. Overlap prevents important context from being cut in half.

## `app/rag/text_utils.py`

Purpose: tokenization and similarity.

```python
STOPWORDS = {...}
```

Stopwords are common words that usually carry little retrieval meaning.

```python
def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
```

This lowercases text, extracts alphanumeric tokens, and removes stopwords by default.

```python
def cosine_similarity(left, right):
```

This compares two sparse vectors. A score near 1 means similar direction. A score near 0 means little overlap.

## `app/rag/vector_store.py`

Purpose: build a local searchable index.

```python
class LocalTfidfIndex:
```

This class behaves like a tiny local vector database.

```python
document_frequency[token] = document_frequency.get(token, 0) + 1
```

This counts how many chunks contain each token.

```python
idf = {
    token: math.log((1 + total_docs) / (1 + frequency)) + 1
}
```

Rare words get higher weight. Common words get lower weight.

```python
def search(self, query: str, top_k: int):
```

Search converts the query into a vector, compares it with every chunk vector, sorts by score, and returns the top results.

## `app/rag/bm25.py`

Purpose: keyword retrieval.

BM25 improves exact matching. It is useful when vector search misses exact terms.

```python
self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
```

BM25 uses average document length to avoid unfairly favoring very long chunks.

```python
score += self._idf(token) * (numerator / denominator)
```

This is the BM25 scoring formula. It rewards query terms that appear in the chunk, especially rare terms.

## `app/rag/retrievers.py`

Purpose: combine retrieval methods.

```python
def reciprocal_rank_fusion(...):
```

RRF merges multiple ranked lists. If a chunk appears high in both vector and BM25 results, it becomes stronger.

```python
class HybridRetriever:
```

This is the main retriever used by the pipeline.

Steps:

1. Run vector search.
2. Run BM25 search.
3. Fuse rankings.
4. Rerank candidates.
5. Return final context chunks.

## `app/rag/prompts.py`

Purpose: build the grounded prompt.

The prompt includes:

- Rules.
- Context blocks.
- User question.
- Answer slot.

This is where hallucination control starts.

## `app/rag/generator.py`

Purpose: turn retrieved chunks into an answer.

```python
class ExtractiveGenerator:
```

This free local generator chooses relevant sentences from retrieved chunks.

```python
class OpenAIGenerator:
```

This optional generator calls an external LLM. It is separated so the app does not require an API key for local learning.

## `app/rag/pipeline.py`

Purpose: connect everything.

```python
def build_index(settings: Settings) -> LocalTfidfIndex:
```

This loads documents, chunks them, builds the index, and saves it.

```python
class RAGPipeline:
```

This is the main application object.

```python
def ask(self, question: str) -> RagAnswer:
```

This is the full RAG call:

1. Retrieve relevant chunks.
2. Generate answer.
3. Return answer plus retrieval details.

## `app/agents/simple_graph.py`

Purpose: show agent-style routing.

```python
def route(state: AgentState) -> AgentState:
```

This decides which tool path to use:

- Calculator.
- Knowledge base.
- Direct answer.

This is the same idea as a graph node in LangGraph.

```python
def act(state: AgentState, pipeline: RAGPipeline) -> AgentState:
```

This calls the selected tool.

```python
def respond(state: AgentState) -> AgentState:
```

This formats the final answer.

## `app/api/main.py`

Purpose: expose the system over HTTP.

Endpoints:

- `GET /health`
- `POST /admin/reindex`
- `POST /ask`
- `POST /agent`

This turns the project from a script into a deployable backend.

## `app/evaluation/evaluate.py`

Purpose: test quality.

The eval set checks if expected terms appear in the answer.

This is simple, but it teaches the production habit: never rely only on manual testing.

