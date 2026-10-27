import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"
    index_path: Path = PROJECT_ROOT / "storage" / "rag_index.json"
    chunk_size_words: int = 120
    chunk_overlap_words: int = 30
    top_k: int = 6
    rerank_top_k: int = 4
    min_score: float = 0.01
    openai_model: str = os.getenv("OPENAI_MODEL", "")


def load_settings() -> Settings:
    return Settings(
        docs_dir=Path(os.getenv("DOCS_DIR", Settings.docs_dir)),
        index_path=Path(os.getenv("INDEX_PATH", Settings.index_path)),
        chunk_size_words=int(os.getenv("CHUNK_SIZE_WORDS", Settings.chunk_size_words)),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", Settings.chunk_overlap_words)),
        top_k=int(os.getenv("TOP_K", Settings.top_k)),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", Settings.rerank_top_k)),
        min_score=float(os.getenv("MIN_SCORE", Settings.min_score)),
        openai_model=os.getenv("OPENAI_MODEL", Settings.openai_model),
    )

