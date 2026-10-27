from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from app.ingestion.models import Chunk
from app.rag.bm25 import BM25Retriever
from app.rag.text_utils import tokenize
from app.rag.vector_store import LocalTfidfIndex


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    source: str


def reciprocal_rank_fusion(result_lists: Iterable[List[Tuple[Chunk, float]]], k: int = 60) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk, _score) in enumerate(results, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
    return fused


def lexical_rerank_score(query: str, chunk: Chunk) -> float:
    query_terms = set(tokenize(query))
    chunk_terms = set(tokenize(chunk.text))
    if not query_terms:
        return 0.0
    overlap = len(query_terms & chunk_terms) / len(query_terms)
    source_bonus = 0.05 if any(term in chunk.metadata.get("file_name", "").lower() for term in query_terms) else 0.0
    return overlap + source_bonus


class HybridRetriever:
    def __init__(self, index: LocalTfidfIndex):
        self.index = index
        self.bm25 = BM25Retriever(index.chunks)

    def search(self, query: str, top_k: int, rerank_top_k: int) -> List[SearchResult]:
        vector_results = self.index.search(query, top_k=top_k * 3)
        keyword_results = self.bm25.search(query, top_k=top_k * 3)
        fused_scores = reciprocal_rank_fusion([vector_results, keyword_results])

        candidates = [
            (self.index.chunk_by_id[chunk_id], score)
            for chunk_id, score in fused_scores.items()
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[: max(top_k * 2, rerank_top_k)]

        reranked = []
        for chunk, fused_score in candidates:
            rerank_score = lexical_rerank_score(query, chunk)
            final_score = fused_score + rerank_score
            reranked.append(SearchResult(chunk=chunk, score=final_score, source="hybrid_rrf_rerank"))

        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:rerank_top_k]

