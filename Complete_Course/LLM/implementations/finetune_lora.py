"""
PEFT / LoRA (Parameter-Efficient Fine-Tuning) configuration and training script.
This script demonstrates how to configure LoRA adapters on top of a base LLM (like Llama/GPT-2)
using Hugging Face `peft` and `transformers`.

Run: python Complete_Course/LLM/implementations/finetune_lora.py
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
# We import PEFT components
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def main():
    if not PEFT_AVAILABLE:
        print("[WARNING] 'peft' library is not installed. Run 'pip install peft bitsandbytes' to use LoRA.")
        print("This script will demonstrate the code layout but will simulate training config.")
        
    model_name = "gpt2"  # Using small GPT-2 as a demo model
    
    # 1. Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load model (for real LLMs, you would pass device_map="auto" and load_in_8bit/4bit=True)
    print(f"Loading base model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 3. Configure LoRA adapter
    # Rank (r) determines size of low-rank matrices. Standard: 8 or 16.
    # Alpha determines scaling factor. Standard: 16 or 32.
    # Target modules specify which layers to apply adapters to (e.g., c_attn for GPT-2, q_proj, v_proj for Llama).
    if PEFT_AVAILABLE:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]  # c_attn is the attention projection matrix in GPT-2
        )
        
        # 4. Wrap base model with LoRA adapters
        model = get_peft_model(model, peft_config)
        print("\n--- LoRA Parameter Breakdown ---")
        model.print_trainable_parameters()
    else:
        print("\n[Simulating PEFT Config]:")
        print("   r = 8 (Rank of projection matrices)")
        print("   lora_alpha = 32 (Scaling multiplier)")
        print("   target_modules = ['c_attn'] (Attention layers to adapt)")
        print("   -> Reduces trainable parameters by ~99%!")
        
    # 5. Load and process dataset (e.g. instruction tuning dataset)
    print("\nLoading dataset (Wikitext dummy subset)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50]")  # small slice for demo

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data collator handles padding batches dynamically
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./lora_results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10,
        logging_steps=5,
        learning_rate=2e-4,  # LoRA parameters tolerate higher learning rates than full weights
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # use FP16 training if GPU is available
        report_to="none"
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
    )

    print("\nReady to train. (In a real pipeline, running 'trainer.train()' would optimize the adapters).")
    # For speed and safety in this sandbox, we don't start the actual GPU-intensive training.
    # trainer.train()

if __name__ == '__main__':
    main()
