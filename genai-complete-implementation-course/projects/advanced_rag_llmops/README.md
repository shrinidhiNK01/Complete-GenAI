# Advanced RAG LLMOps Project

This is a complete teaching project for an end-to-end Generative AI application.

It includes:

- Document ingestion.
- Text cleaning.
- Chunking with overlap.
- Local TF-IDF vector index.
- BM25 keyword retrieval.
- Hybrid retrieval.
- Reciprocal rank fusion.
- Lightweight reranking.
- Grounded answer generation.
- Simple agent routing.
- FastAPI endpoints.
- Evaluation script.
- Unit tests.
- Docker.
- GitHub Actions CI.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cli.py ingest
python cli.py ask "What is the refund window?"
python cli.py agent "calculate 2 + 3 * 4"
python cli.py eval
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python cli.py ingest
python cli.py ask "What is the refund window?"
python cli.py agent "calculate 2 + 3 * 4"
python cli.py eval
```

## Run API

```bash
uvicorn app.api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## Docker

```bash
docker build -t advanced-rag-llmops .
docker run -p 8000:8000 advanced-rag-llmops
```

