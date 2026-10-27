import csv
import json
from pathlib import Path
from typing import Iterable, List

from app.ingestion.models import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json"}


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_csv_file(path: Path) -> str:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            parts = [f"{key}: {value}" for key, value in row.items()]
            rows.append(f"Row {row_number}. " + ". ".join(parts))
    return "\n".join(rows)


def load_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return json.dumps(data, ensure_ascii=True, indent=2)


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = load_text_file(path)
    elif suffix == ".csv":
        text = load_csv_file(path)
    elif suffix == ".json":
        text = load_json_file(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return Document(
        doc_id=path.stem,
        text=text,
        metadata={"source": str(path), "file_name": path.name, "suffix": suffix},
    )


def iter_document_paths(docs_dir: Path) -> Iterable[Path]:
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_documents(docs_dir: Path) -> List[Document]:
    return [load_document(path) for path in iter_document_paths(docs_dir)]

