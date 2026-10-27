from pathlib import Path
from typing import List, Tuple


class PolicyKnowledgeBase:
    def __init__(self, policy_dir: Path):
        self.documents = []
        for path in sorted(policy_dir.glob("*.txt")):
            self.documents.append((path.name, path.read_text(encoding="utf-8")))

    def search(self, query: str, limit: int = 2) -> List[Tuple[str, str]]:
        terms = set(query.lower().split())
        scored = []
        for name, text in self.documents:
            lowered = text.lower()
            score = sum(1 for term in terms if term in lowered)
            if score:
                scored.append((score, name, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [(name, text) for _score, name, text in scored[:limit]]

    def answer(self, query: str) -> Tuple[str, List[str]]:
        results = self.search(query)
        if not results:
            return "I do not know from the policy documents.", []
        answer_parts = []
        sources = []
        for name, text in results:
            first_sentence = text.split(".")[0].strip()
            answer_parts.append(first_sentence + ".")
            sources.append(name)
        return " ".join(answer_parts), sources

