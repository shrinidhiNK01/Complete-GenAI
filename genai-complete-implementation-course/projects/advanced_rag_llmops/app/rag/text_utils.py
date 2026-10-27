import math
import re
from collections import Counter
from typing import Dict, Iterable, List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "with", "what", "when", "where", "who", "why", "how", "do", "does",
}


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def term_counts(tokens: Iterable[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)

