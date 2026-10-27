from typing import Iterable

from app.rag.retrievers import SearchResult


def build_grounded_prompt(question: str, results: Iterable[SearchResult]) -> str:
    context_blocks = []
    for number, result in enumerate(results, start=1):
        source = result.chunk.metadata.get("file_name", result.chunk.doc_id)
        context_blocks.append(
            f"[Source {number}: {source} | chunk_id={result.chunk.chunk_id}]\n{result.chunk.text}"
        )

    context = "\n\n".join(context_blocks)
    return f"""You are a careful RAG assistant.

Rules:
1. Answer only using the context.
2. If the context does not contain the answer, say: "I do not know from the provided context."
3. Mention the source file names used.
4. Keep the answer clear and simple.

Context:
{context}

Question:
{question}

Answer:
"""

