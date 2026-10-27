# 06. Fine-Tuning, Alignment, LoRA, QLoRA, Distillation, and Quantization

## When to Use Fine-Tuning

Use fine-tuning for:

- Style.
- Format.
- Domain-specific behavior.
- Classification.
- Extraction.
- Small-model specialization.

Do not use fine-tuning just to add frequently changing knowledge. Use RAG for that.

## RAG vs Fine-Tuning vs Agents

| Need | Use |
| --- | --- |
| Private documents | RAG |
| Fresh knowledge | RAG |
| Stable style | Fine-tuning |
| Strict output format | Fine-tuning or structured output |
| Tools/actions | Agent |
| Multi-step workflow | Agent |
| Smaller cheaper model | Fine-tuning/distillation |

## Dataset Format

Classification example:

```json
{
  "input": "I cannot reset my password.",
  "output": "account_access"
}
```

Instruction example:

```json
{
  "instruction": "Extract invoice fields.",
  "input": "Invoice INV-100 total $500 due tomorrow.",
  "output": {
    "invoice_number": "INV-100",
    "amount": 500,
    "due_date": "tomorrow"
  }
}
```

Preference example:

```json
{
  "prompt": "Explain RAG.",
  "chosen": "RAG retrieves relevant documents and uses them to answer.",
  "rejected": "RAG is random text generation."
}
```

## Fine-Tuning Procedure

1. Define task.
2. Collect examples.
3. Clean data.
4. Remove private data if not allowed.
5. Split train/validation/test.
6. Choose model.
7. Choose method: full fine-tuning, LoRA, QLoRA.
8. Train.
9. Evaluate.
10. Compare to baseline prompting/RAG.
11. Deploy.
12. Monitor drift.

## LoRA

LoRA trains small adapter matrices instead of all weights.

Benefits:

- Cheaper.
- Lower memory.
- Faster experiments.
- Many adapters can share one base model.

## QLoRA

QLoRA loads the base model in low precision and trains LoRA adapters.

Use when GPU memory is limited.

## Full Fine-Tuning

Full fine-tuning updates all weights.

Use only when:

- You have strong data.
- You have enough compute.
- PEFT is not enough.
- You can evaluate thoroughly.

## Hugging Face Fine-Tuning Shape

Typical steps:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModelForSequenceClassification.from_pretrained("model-name", num_labels=3)
```

Explanation:

- `load_dataset` loads data.
- `AutoTokenizer` converts text to tokens.
- `AutoModelForSequenceClassification` loads a model head for classification.
- `Trainer` handles training loop.

## Distillation

Distillation trains a smaller student model from a larger teacher model.

Use when:

- Large model is too expensive.
- Task is narrow.
- You need lower latency.

## Quantization

Quantization reduces precision.

Examples:

- FP32: accurate but large.
- FP16/BF16: common for GPU inference.
- INT8: smaller.
- INT4: much smaller.

Tradeoff:

- Lower memory and faster inference.
- Possible quality loss.

## RLHF

RLHF steps:

1. Generate answers.
2. Humans rank answers.
3. Train reward model.
4. Optimize model using reinforcement learning.

## DPO

DPO trains directly from chosen/rejected pairs.

Benefits:

- Simpler than RLHF.
- No separate reward model.
- Strong practical alignment method.

## ORPO and PPO

PPO is a reinforcement learning algorithm often associated with RLHF.

ORPO is another preference optimization approach that tries to combine supervised learning and preference optimization more directly.

## Embedding Fine-Tuning

Fine-tune embeddings when retrieval fails for domain-specific meaning.

Training sample:

```json
{
  "query": "SLA violation",
  "positive": "The service level agreement was breached.",
  "negative": "The product was shipped yesterday."
}
```

Goal:

The query should be closer to the positive document than the negative document.

## Fine-Tuning Evaluation

Check:

- Accuracy.
- F1.
- Exact format match.
- Human preference.
- Safety regressions.
- Latency.
- Cost.
- Out-of-domain behavior.

