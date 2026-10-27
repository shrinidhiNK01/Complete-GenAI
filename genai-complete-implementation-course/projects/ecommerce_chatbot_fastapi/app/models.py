from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ChatRequest:
    user_id: str
    message: str


@dataclass
class ChatResponse:
    answer: str
    route: str
    sources: List[str]
    needs_human_review: bool = False
    metadata: Dict[str, Any] | None = None

