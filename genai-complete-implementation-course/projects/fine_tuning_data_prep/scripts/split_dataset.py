import argparse
import json
import random
from pathlib import Path

try:
    from scripts.validate_dataset import load_jsonl, validate_dataset
except ModuleNotFoundError:
    from validate_dataset import load_jsonl, validate_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def split_dataset(input_path: Path, output_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> dict:
    validation = validate_dataset(input_path)
    if not validation["valid"]:
        return {"valid": False, "errors": validation["errors"]}

    rows = load_jsonl(input_path)
    random.Random(seed).shuffle(rows)

    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)

    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    for split_name, split_rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows)

    return {
        "valid": True,
        "total": len(rows),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split fine-tuning JSONL data")
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = split_dataset(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
