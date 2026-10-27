from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.core.config import Settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_documents
from app.rag.generator import Generator, build_generator
from app.rag.retrievers import HybridRetriever, SearchResult
from app.rag.vector_store import LocalTfidfIndex


@dataclass
class RagAnswer:
    question: str
    answer: str
    results: List[SearchResult]


def build_index(settings: Settings) -> LocalTfidfIndex:
    documents = load_documents(settings.docs_dir)
    chunks = chunk_documents(
        documents,
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    index = LocalTfidfIndex.build(chunks)
    index.save(settings.index_path)
    return index


def load_or_build_index(settings: Settings) -> LocalTfidfIndex:
    if Path(settings.index_path).exists():
        return LocalTfidfIndex.load(settings.index_path)
    return build_index(settings)


class RAGPipeline:
    def __init__(self, settings: Settings, generator: Generator | None = None):
        self.settings = settings
        self.index = load_or_build_index(settings)
        self.retriever = HybridRetriever(self.index)
        self.generator = generator or build_generator()

    def ask(self, question: str) -> RagAnswer:
        results = self.retriever.search(
            question,
            top_k=self.settings.top_k,
            rerank_top_k=self.settings.rerank_top_k,
        )
        answer = self.generator.generate(question, results)
        return RagAnswer(question=question, answer=answer, results=results)

