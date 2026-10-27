import argparse
import json
from pathlib import Path
from typing import Dict, List


REQUIRED_FIELDS = {"instruction", "input", "output"}


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number} is not valid JSON: {exc}") from exc
            rows.append(row)
    return rows


def validate_row(row: Dict, row_number: int) -> List[str]:
    errors = []
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        errors.append(f"row {row_number}: missing fields {sorted(missing)}")
    for field in REQUIRED_FIELDS & set(row):
        if not isinstance(row[field], str) or not row[field].strip():
            errors.append(f"row {row_number}: {field} must be a non-empty string")
    return errors


def validate_dataset(path: Path) -> Dict:
    rows = load_jsonl(path)
    errors = []
    seen = set()
    duplicates = 0

    for index, row in enumerate(rows, start=1):
        errors.extend(validate_row(row, index))
        fingerprint = (
            row.get("instruction", "").strip().lower(),
            row.get("input", "").strip().lower(),
            row.get("output", "").strip().lower(),
        )
        if fingerprint in seen:
            duplicates += 1
        seen.add(fingerprint)

    return {
        "path": str(path),
        "rows": len(rows),
        "duplicates": duplicates,
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate instruction fine-tuning JSONL data")
    parser.add_argument("path")
    args = parser.parse_args()
    report = validate_dataset(Path(args.path))
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

