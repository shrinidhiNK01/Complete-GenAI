import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from app.ingestion.models import Chunk
from app.rag.text_utils import cosine_similarity, term_counts, tokenize


class LocalTfidfIndex:
    def __init__(self, chunks: List[Chunk], idf: Dict[str, float], vectors: Dict[str, Dict[str, float]]):
        self.chunks = chunks
        self.idf = idf
        self.vectors = vectors
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "LocalTfidfIndex":
        document_frequency: Dict[str, int] = {}
        tokenized_chunks: Dict[str, List[str]] = {}

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tokenized_chunks[chunk.chunk_id] = tokens
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total_docs = max(len(chunks), 1)
        idf = {
            token: math.log((1 + total_docs) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }

        vectors = {
            chunk_id: cls._tfidf_vector(tokens, idf)
            for chunk_id, tokens in tokenized_chunks.items()
        }

        return cls(chunks=chunks, idf=idf, vectors=vectors)

    @staticmethod
    def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        counts = term_counts(tokens)
        total = max(sum(counts.values()), 1)
        return {
            token: (count / total) * idf.get(token, 0.0)
            for token, count in counts.items()
        }

    def query_vector(self, query: str) -> Dict[str, float]:
        return self._tfidf_vector(tokenize(query), self.idf)

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        query_vector = self.query_vector(query)
        scored = []
        for chunk in self.chunks:
            score = cosine_similarity(query_vector, self.vectors.get(chunk.chunk_id, {}))
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "idf": self.idf,
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalTfidfIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**chunk_data) for chunk_data in payload["chunks"]]
        return cls(chunks=chunks, idf=payload["idf"], vectors=payload["vectors"])

