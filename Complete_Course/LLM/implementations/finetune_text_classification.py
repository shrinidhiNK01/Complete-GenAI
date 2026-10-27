"""
Minimal fine-tuning script for text classification using Hugging Face Trainer.

This is a starting point — adapt dataset and hyperparameters for real tasks.
"""
# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }


def main():
    # Example uses SST2 subset from GLUE for demo purposes
    dataset = load_dataset('glue', 'sst2')
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['sentence'], truncation=True, padding='max_length', max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(['sentence', 'idx'])
    tokenized.set_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()
