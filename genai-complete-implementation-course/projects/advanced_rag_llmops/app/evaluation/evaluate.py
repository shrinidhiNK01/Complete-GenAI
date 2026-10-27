import json
from pathlib import Path

from app.core.config import load_settings
from app.evaluation.metrics import contains_expected_terms, has_source
from app.rag.pipeline import RAGPipeline


EVAL_SET = [
    {
        "question": "What is the refund window?",
        "expected_terms": ["30 days", "proof of purchase"],
    },
    {
        "question": "How long does standard shipping take?",
        "expected_terms": ["3 to 5 business days"],
    },
    {
        "question": "What should production RAG systems log?",
        "expected_terms": ["request id", "latency", "retrieval"],
    },
]


def run_evaluation(output_path: Path | None = None) -> dict:
    pipeline = RAGPipeline(load_settings())
    rows = []
    for case in EVAL_SET:
        result = pipeline.ask(case["question"])
        term_score = contains_expected_terms(result.answer, case["expected_terms"])
        rows.append(
            {
                "question": case["question"],
                "answer": result.answer,
                "term_score": term_score,
                "has_source": has_source(result.answer),
            }
        )

    average_score = sum(row["term_score"] for row in rows) / max(len(rows), 1)
    report = {"average_term_score": average_score, "rows": rows}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), ensure_ascii=True, indent=2))

