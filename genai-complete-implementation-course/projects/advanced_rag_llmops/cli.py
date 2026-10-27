import argparse
import json

from app.agents.simple_graph import run_agent
from app.core.config import load_settings
from app.core.logging import configure_logging
from app.evaluation.evaluate import run_evaluation
from app.rag.pipeline import RAGPipeline, build_index


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Advanced RAG LLMOps teaching CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build the local RAG index")

    ask_parser = subparsers.add_parser("ask", help="Ask the RAG system a question")
    ask_parser.add_argument("question")

    agent_parser = subparsers.add_parser("agent", help="Ask the simple agent a question")
    agent_parser.add_argument("question")

    subparsers.add_parser("eval", help="Run the small evaluation set")

    args = parser.parse_args()
    settings = load_settings()

    if args.command == "ingest":
        index = build_index(settings)
        print(f"Indexed {len(index.chunks)} chunks at {settings.index_path}")
    elif args.command == "ask":
        pipeline = RAGPipeline(settings)
        result = pipeline.ask(args.question)
        print(result.answer)
    elif args.command == "agent":
        pipeline = RAGPipeline(settings)
        state = run_agent(args.question, pipeline)
        print(state.final_answer)
    elif args.command == "eval":
        print(json.dumps(run_evaluation(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

