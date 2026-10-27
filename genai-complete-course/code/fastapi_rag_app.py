import os
from typing import List

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS = [
    "Refunds are allowed within 30 days of purchase with proof of purchase.",
    "Standard shipping takes 3 to 5 business days.",
    "Enterprise customers receive 24 hour premium support.",
]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    context: List[str]


app = FastAPI(title="Minimal RAG API")
client = OpenAI()


def retrieve(question: str, top_k: int = 2) -> List[str]:
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(DOCS)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]
    ranked_indexes = scores.argsort()[::-1][:top_k]
    return [DOCS[index] for index in ranked_indexes]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    context = retrieve(request.question)
    prompt = f"""
Answer the question using only the context.
If the context is not enough, say you do not know.

Context:
{chr(10).join(context)}

Question:
{request.question}
"""
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        instructions="You are a careful RAG assistant.",
        input=prompt,
    )
    return AskResponse(answer=response.output_text, context=context)

