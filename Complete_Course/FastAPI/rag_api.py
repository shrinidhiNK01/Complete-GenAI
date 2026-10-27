"""
Advanced RAG API exposing vector search, re-ranking, and streaming responses.
This script demonstrates:
1. Document ingestion: indexing chunks into a vector store.
2. Advanced retrieval: search + Cross-Encoder re-ranking.
3. Chat Streaming: using FastAPI StreamingResponse to stream answers token-by-token.

Run: uvicorn Complete_Course.FastAPI.rag_api:app --reload
"""

import sys
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add RAG directory to path to import vector store and pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent / 'RAG' / 'implementations'))
try:
    # pyrefly: ignore [missing-import]
    from advanced_rag_pipeline import AdvancedRAGPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

app = FastAPI(
    title="RAG API & Streaming Masterclass",
    description="Exposes advanced RAG and streaming text endpoints",
    version="1.0.0"
)

# Initialize global RAG pipeline
if PIPELINE_AVAILABLE:
    rag_pipeline = AdvancedRAGPipeline()
    # Ingest some seed data on startup
    sample_text = (
        "Study Pack Support: For questions regarding this ML/DL course, contact support@studypack.com. "
        "Our support hours are Monday to Friday from 9 AM to 5 PM EST. "
        "We promise response times under 24 hours. "
        "FastAPI is a Python framework that uses Starlette and Pydantic for building APIs. "
        "LangGraph is built by LangChain to create stateful multi-agent systems with cyclic execution graphs."
    )
    rag_pipeline.ingest_document(sample_text, metadata={"source": "faq"})
else:
    rag_pipeline = None
    print("[WARNING] Could not import AdvancedRAGPipeline. Place pipeline files correctly.")

# --- Data Schemas ---
class DocumentIn(BaseModel):
    text: str
    source_name: str = "custom_upload"

class QueryIn(BaseModel):
    question: str
    k_vector: int = 5
    k_final: int = 2

# --- Routes ---
@app.post("/ingest")
def ingest_document(doc: DocumentIn):
    """Ingest raw text, chunk it, create embeddings, and index it in the vector store."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not available.")
    try:
        rag_pipeline.ingest_document(doc.text, metadata={"source": doc.source_name})
        return {"status": "success", "message": f"Successfully ingested document from '{doc.source_name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_rag(req: QueryIn):
    """Retrieve relevant chunks, re-rank them, and return context."""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not available.")
        
    # Retrieve and re-rank context chunks
    candidates = rag_pipeline.retrieve_and_rerank(
        query=req.question,
        k_vector=req.k_vector,
        k_final=req.k_final
    )
    
    context_chunks = [c["document"] for c in candidates]
    context_str = "\n".join(context_chunks)
    
    # Compile prompt
    prompt = (
        "Context:\n"
        f"{context_str}\n\n"
        f"Question: {req.question}\n"
        "Answer: "
    )
    
    # Simulate a generation answer (or call real LLM)
    simulated_answer = (
        f"Based on the FAQ files, here is the answer: "
        f"For course questions, email support@studypack.com during business hours. "
        f"FastAPI and LangGraph are key frameworks covered in this pack."
    )
    
    return {
        "question": req.question,
        "answer": simulated_answer,
        "retrieved_context": context_chunks,
        "raw_prompt": prompt
    }


@app.post("/query/stream")
def query_rag_stream(req: QueryIn):
    """
    Query the RAG pipeline and stream the generated answer token-by-token
    to the client using HTTP chunked transfer encoding (StreamingResponse).
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG Pipeline not available.")

    # 1. Retrieve relevant info
    candidates = rag_pipeline.retrieve_and_rerank(
        query=req.question,
        k_vector=req.k_vector,
        k_final=req.k_final
    )
    context_str = "\n".join([c["document"] for c in candidates])
    
    # 2. Simulated LLM generation generator
    def response_generator():
        # Yield metadata first (e.g. source documents)
        yield "event: sources\ndata: " + str([c["metadata"].get("source", "unknown") for c in candidates]) + "\n\n"
        time.sleep(0.5)
        
        full_answer = (
            f"Here is the retrieved answer regarding your question '{req.question}':\n\n"
            f"According to the FAQ manual, support is available at support@studypack.com. "
            f"FastAPI is built on top of Starlette and Pydantic, and LangGraph is used to build stateful multi-agent systems."
        )
        
        # We split the sentence by words and yield them one by one to simulate token streaming
        words = full_answer.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            time.sleep(0.08)  # simulate model latency
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")
