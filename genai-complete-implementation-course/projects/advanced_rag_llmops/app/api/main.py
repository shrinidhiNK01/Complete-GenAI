from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.simple_graph import run_agent
from app.core.config import load_settings
from app.core.logging import configure_logging
from app.rag.pipeline import RAGPipeline, build_index


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


configure_logging()
settings = load_settings()
pipeline = RAGPipeline(settings)
app = FastAPI(title="Advanced RAG LLMOps Teaching API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/reindex")
def reindex():
    build_index(settings)
    return {"status": "reindexed", "index_path": str(settings.index_path)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = pipeline.ask(request.question)
    sources = [item.chunk.metadata.get("file_name", item.chunk.doc_id) for item in result.results]
    return AskResponse(answer=result.answer, sources=sorted(set(sources)))


@app.post("/agent", response_model=AskResponse)
def agent(request: AskRequest):
    state = run_agent(request.question, pipeline)
    return AskResponse(answer=state.final_answer, sources=[state.route])

