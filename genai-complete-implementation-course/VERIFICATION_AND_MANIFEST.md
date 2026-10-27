# Verification and Manifest

## Main Folder

```text
outputs/genai-complete-implementation-course
```

## Current Files

Documentation:

- `README.md`
- `VERIFICATION_AND_MANIFEST.md`
- `docs/10_PROJECT_ECOMMERCE_CHATBOT_END_TO_END.md`
- `docs/11_PROJECT_MULTIMODAL_RAG_END_TO_END.md`
- `docs/12_PROJECT_FINE_TUNING_DATA_PIPELINE.md`
- `docs/13_DEPLOYMENT_COOKBOOK_MULTIPLE_PROJECTS.md`

Projects:

- `projects/ecommerce_chatbot_fastapi`
- `projects/fine_tuning_data_prep`

## Verification Status

Latest verification completed:

### E-Commerce Chatbot

Project:

```text
projects/ecommerce_chatbot_fastapi
```

Commands:

```bash
python -m compileall app cli.py
python cli.py "recommend running shoe"
python cli.py "what is the refund policy"
python cli.py "track order ORD1001" --user-id user_1
python -m pytest
```

Results:

- Python compilation passed.
- Product search route worked.
- Policy RAG route worked.
- Authorized order lookup worked.
- Tests passed: `3 passed`.

### Fine-Tuning Data Prep

Project:

```text
projects/fine_tuning_data_prep
```

Commands:

```bash
python -m compileall scripts
python scripts/validate_dataset.py data/sample_instruction_dataset.jsonl
python scripts/estimate_tokens.py data/sample_instruction_dataset.jsonl
python scripts/split_dataset.py data/sample_instruction_dataset.jsonl outputs/splits
python -m pytest
```

Results:

- Python compilation passed.
- Dataset validation passed: `valid: true`.
- Rough token estimate completed: `342` total rough tokens.
- Dataset split completed: `8 train`, `1 validation`, `1 test`.
- Tests passed: `2 passed`.
