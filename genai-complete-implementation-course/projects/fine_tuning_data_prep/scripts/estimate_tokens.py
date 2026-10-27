import argparse
import json
import re
from pathlib import Path

try:
    from scripts.validate_dataset import load_jsonl
except ModuleNotFoundError:
    from validate_dataset import load_jsonl


def rough_token_count(text: str) -> int:
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(pieces) * 1.25))


def estimate_dataset_tokens(path: Path) -> dict:
    rows = load_jsonl(path)
    row_reports = []
    total = 0
    for index, row in enumerate(rows, start=1):
        combined = "\n".join(
            [
                row.get("instruction", ""),
                row.get("input", ""),
                row.get("output", ""),
            ]
        )
        tokens = rough_token_count(combined)
        total += tokens
        row_reports.append({"row": index, "rough_tokens": tokens})
    return {
        "path": str(path),
        "rows": len(rows),
        "rough_total_tokens": total,
        "rough_average_tokens": total / max(len(rows), 1),
        "rows_detail": row_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate rough token counts for JSONL fine-tuning data")
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(estimate_dataset_tokens(Path(args.path)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
