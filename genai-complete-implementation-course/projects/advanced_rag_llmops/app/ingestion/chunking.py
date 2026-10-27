import re
from typing import List

from app.ingestion.models import Chunk, Document


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def chunk_document(document: Document, chunk_size_words: int, overlap_words: int) -> List[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = split_words(clean_text(document.text))
    chunks: List[Chunk] = []
    start = 0
    chunk_number = 1

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunk_id = f"{document.doc_id}:chunk:{chunk_number}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=chunk_text,
                start_word=start,
                end_word=end,
                metadata=document.metadata.copy(),
            )
        )
        if end == len(words):
            break
        start = end - overlap_words
        chunk_number += 1

    return chunks


def chunk_documents(documents: List[Document], chunk_size_words: int, overlap_words: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size_words, overlap_words))
    return chunks

