import math
from typing import Dict, List, Tuple

from app.ingestion.models import Chunk
from app.rag.text_utils import term_counts, tokenize


class BM25Retriever:
    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokenized]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.doc_frequency = self._document_frequency()

    def _document_frequency(self) -> Dict[str, int]:
        frequency: Dict[str, int] = {}
        for tokens in self.tokenized:
            for token in set(tokens):
                frequency[token] = frequency.get(token, 0) + 1
        return frequency

    def _idf(self, token: str) -> float:
        total_docs = len(self.chunks)
        docs_with_token = self.doc_frequency.get(token, 0)
        return math.log(1 + (total_docs - docs_with_token + 0.5) / (docs_with_token + 0.5))

    def score(self, query: str, index: int) -> float:
        query_tokens = tokenize(query)
        document_tokens = self.tokenized[index]
        counts = term_counts(document_tokens)
        document_length = self.lengths[index]
        score = 0.0

        for token in query_tokens:
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (1 - self.b + self.b * document_length / max(self.avg_length, 1))
            score += self._idf(token) * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        scored = [(chunk, self.score(query, index)) for index, chunk in enumerate(self.chunks)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

