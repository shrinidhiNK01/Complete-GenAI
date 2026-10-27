import argparse
from dataclasses import asdict
from pathlib import Path

from app.models import ChatRequest
from app.services.router import EcommerceAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce chatbot CLI")
    parser.add_argument("message")
    parser.add_argument("--user-id", default="user_1")
    args = parser.parse_args()

    assistant = EcommerceAssistant(Path(__file__).resolve().parent / "data")
    response = assistant.chat(ChatRequest(user_id=args.user_id, message=args.message))
    print(asdict(response))


if __name__ == "__main__":
    main()

