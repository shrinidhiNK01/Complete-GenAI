from pathlib import Path

from scripts.estimate_tokens import estimate_dataset_tokens
from scripts.validate_dataset import validate_dataset


DATASET = Path(__file__).resolve().parents[1] / "data" / "sample_instruction_dataset.jsonl"


def test_sample_dataset_is_valid():
    report = validate_dataset(DATASET)
    assert report["valid"] is True
    assert report["rows"] == 10


def test_token_estimate_has_total():
    report = estimate_dataset_tokens(DATASET)
    assert report["rough_total_tokens"] > 0
    assert report["rough_average_tokens"] > 0

