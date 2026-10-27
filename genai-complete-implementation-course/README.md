# Generative AI Complete Implementation Course

This course folder is being built as a practical beginner-to-advanced Generative AI implementation pack.

Current runnable projects:

- `projects/ecommerce_chatbot_fastapi`
- `projects/fine_tuning_data_prep`

Current deep project docs:

- `docs/10_PROJECT_ECOMMERCE_CHATBOT_END_TO_END.md`
- `docs/11_PROJECT_MULTIMODAL_RAG_END_TO_END.md`
- `docs/12_PROJECT_FINE_TUNING_DATA_PIPELINE.md`
- `docs/13_DEPLOYMENT_COOKBOOK_MULTIPLE_PROJECTS.md`

## Project 1: E-Commerce Chatbot

This project teaches:

- Product search.
- Policy RAG.
- Order lookup.
- User authorization.
- Guardrails.
- FastAPI.
- Docker.
- Tests.

Run:

```bash
cd projects/ecommerce_chatbot_fastapi
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cli.py "recommend running shoe"
python cli.py "what is the refund policy"
python cli.py "track order ORD1001" --user-id user_1
pytest
```

Windows PowerShell:

```powershell
cd projects\ecommerce_chatbot_fastapi
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python cli.py "recommend running shoe"
python cli.py "what is the refund policy"
python cli.py "track order ORD1001" --user-id user_1
pytest
```

## Project 2: Fine-Tuning Data Prep

This project teaches:

- Instruction dataset format.
- JSONL validation.
- Duplicate checks.
- Train/validation/test split.
- Rough token estimation.
- Fine-tuning data quality workflow.

Run:

```bash
cd projects/fine_tuning_data_prep
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
cd projects\fine_tuning_data_prep
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\validate_dataset.py data\sample_instruction_dataset.jsonl
python scripts\estimate_tokens.py data\sample_instruction_dataset.jsonl
python scripts\split_dataset.py data\sample_instruction_dataset.jsonl outputs\splits
pytest
```

## What Still Fits Into This Course

The existing docs also explain multimodal RAG and multi-cloud deployment. A full advanced RAG/agent project can be added as the next continuation layer if you want this folder to keep growing project-by-project.

