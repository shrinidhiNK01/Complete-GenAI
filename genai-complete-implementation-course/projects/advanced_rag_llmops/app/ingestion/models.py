from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_word: int
    end_word: int
    metadata: Dict[str, str] = field(default_factory=dict)

