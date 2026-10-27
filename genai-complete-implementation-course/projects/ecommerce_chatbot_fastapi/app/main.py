from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.models import ChatRequest
from app.services.router import EcommerceAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
assistant = EcommerceAssistant(PROJECT_ROOT / "data")
app = FastAPI(title="E-commerce Chatbot Teaching API")


class ChatPayload(BaseModel):
    user_id: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatPayload):
    response = assistant.chat(ChatRequest(user_id=payload.user_id, message=payload.message))
    return asdict(response)

