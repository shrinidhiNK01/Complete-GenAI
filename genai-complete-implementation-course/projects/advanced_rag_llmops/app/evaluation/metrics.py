from typing import Iterable


def contains_expected_terms(answer: str, expected_terms: Iterable[str]) -> float:
    terms = list(expected_terms)
    if not terms:
        return 1.0
    lowered = answer.lower()
    matched = sum(1 for term in terms if term.lower() in lowered)
    return matched / len(terms)


def has_source(answer: str) -> bool:
    return "sources:" in answer.lower() or "source:" in answer.lower()

