# Fine-Tuning Data Preparation Project

Fine-tuning usually fails because the data is weak, inconsistent, duplicated, or poorly formatted. This project teaches the data preparation stage before any expensive training.

## What It Teaches

- JSONL instruction dataset format.
- Required fields.
- Dataset validation.
- Duplicate detection.
- Train/validation/test split.
- Rough token estimation.
- Reproducible data preparation.

## Dataset Format

Each line is one JSON object:

```json
{"instruction":"Classify this support ticket.","input":"I cannot reset my password.","output":"account_access"}
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_dataset.py data/sample_instruction_dataset.jsonl
python scripts/estimate_tokens.py data/sample_instruction_dataset.jsonl
python scripts/split_dataset.py data/sample_instruction_dataset.jsonl outputs/splits
pytest
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\validate_dataset.py data\sample_instruction_dataset.jsonl
python scripts\estimate_tokens.py data\sample_instruction_dataset.jsonl
python scripts\split_dataset.py data\sample_instruction_dataset.jsonl outputs\splits
pytest
```

## Production Upgrades

- Add PII detection.
- Add label consistency checks.
- Add toxicity/safety checks.
- Add train/test leakage detection.
- Add task-specific evaluation sets.
- Export to provider-specific fine-tuning format.

