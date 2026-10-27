import os
import re
from typing import List, Protocol

from app.rag.prompts import build_grounded_prompt
from app.rag.retrievers import SearchResult
from app.rag.text_utils import tokenize


class Generator(Protocol):
    def generate(self, question: str, results: List[SearchResult]) -> str:
        ...


class ExtractiveGenerator:
    def generate(self, question: str, results: List[SearchResult]) -> str:
        query_terms = set(tokenize(question))
        selected_sentences = []
        used_sources = []

        for result in results:
            source = result.chunk.metadata.get("file_name", result.chunk.doc_id)
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            for index, sentence in enumerate(sentences):
                sentence_terms = set(tokenize(sentence))
                if query_terms & sentence_terms:
                    window = sentences[index : index + 2]
                    selected_sentences.extend(item.strip() for item in window if item.strip())
                    used_sources.append(source)
                    break

        if not selected_sentences:
            return "I do not know from the provided context."

        unique_sources = sorted(set(used_sources))
        answer = " ".join(selected_sentences[:4])
        return f"{answer}\n\nSources: {', '.join(unique_sources)}"


class OpenAIGenerator:
    def __init__(self, model: str):
        if not model:
            raise ValueError("OPENAI_MODEL must be set to use OpenAIGenerator")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(self, question: str, results: List[SearchResult]) -> str:
        prompt = build_grounded_prompt(question, results)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text


def build_generator(provider: str = "extractive") -> Generator:
    if provider == "openai":
        return OpenAIGenerator(model=os.getenv("OPENAI_MODEL", ""))
    return ExtractiveGenerator()
