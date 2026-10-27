RAG + FastAPI Example
=====================

This example shows how to build a FAISS index from a folder of text files and
expose a simple FastAPI app that answers questions using retrieved passages
plus a text-generation model.

Requirements:
- Install from `Complete_Course/requirements.txt` (ensure `faiss-cpu` and `sentence-transformers` are installed).

Steps:
1. Prepare documents: place `.md` or `.txt` files under `Complete_Course/RAG/docs/`.
2. Build the index (from the RAG folder):

```bash
python implementations/build_faiss.py --docs_dir ./docs --index_path ../faiss_index.index --data_path ../faiss_data.pkl
```

3. Run the API (from `Complete_Course/FastAPI`):

```bash
uvicorn rag_api:app --reload
```

4. Query the API:

POST http://localhost:8000/query
JSON body: {"question": "What is ...?", "top_k": 5}

Notes:
- The generation model used in this example is small (`gpt2`) for demo. For
  factual QA prefer instruction-tuned or retrieval-augmented models.
- For production, use streaming, batching, and secure API practices.
