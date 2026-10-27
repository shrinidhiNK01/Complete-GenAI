# 12. Project: Fine-Tuning Data Pipeline

Project folder:

```text
projects/fine_tuning_data_prep
```

## Aim

Prepare high-quality fine-tuning data before training.

Fine-tuning is not mainly a model problem. It is mostly a data quality problem.

## Dataset Format

The project uses instruction JSONL:

```json
{"instruction":"Classify this support ticket.","input":"I cannot reset my password.","output":"account_access"}
```

Why JSONL?

- One example per line.
- Easy to stream.
- Easy to validate.
- Common in fine-tuning workflows.

## Step 1: Validate Dataset

Script:

```text
scripts/validate_dataset.py
```

Checks:

- Each line is valid JSON.
- Required fields exist.
- Fields are non-empty strings.
- Duplicate examples are counted.

Run:

```bash
python scripts/validate_dataset.py data/sample_instruction_dataset.jsonl
```

## Step 2: Estimate Tokens

Script:

```text
scripts/estimate_tokens.py
```

Purpose:

- Estimate training size.
- Estimate cost.
- Find very long examples.

Run:

```bash
python scripts/estimate_tokens.py data/sample_instruction_dataset.jsonl
```

Note:

The token estimate is rough. Real providers use their own tokenizer.

## Step 3: Split Dataset

Script:

```text
scripts/split_dataset.py
```

Creates:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

Run:

```bash
python scripts/split_dataset.py data/sample_instruction_dataset.jsonl outputs/splits
```

## Step 4: Training

This project stops before actual training because training depends on provider/framework:

- Hugging Face.
- OpenAI fine-tuning.
- Vertex AI Gemini tuning.
- LLaMA Factory.
- Axolotl.
- Unsloth.

But the data preparation logic is reusable for all of them.

## Step 5: Evaluation

Before training:

- Keep a test set hidden.
- Define exact metrics.
- Build baseline prompt performance.

After training:

- Compare fine-tuned model to baseline.
- Check regressions.
- Check safety.
- Check output format.

## When Fine-Tuning Is Worth It

Fine-tune when:

- Prompting is too inconsistent.
- Output format must be stable.
- Domain tone is important.
- A smaller model must perform a narrow task.

Use RAG instead when:

- The issue is missing knowledge.
- Data changes often.
- Citations are required.

