from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from datasets import Dataset
from synthetic_data import generate_synthetic_data
from evaluate_lora import evaluate_lora

from typing import List, Dict
import os

class SyntheticPrimitiveDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        encoding = self.tokenizer(
            example['input'],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            example['output'],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )['input_ids']
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

def run_phase3_optimized(new_primitives_to_train, base_model_name="llama-7b",
                          library_dir="./primitive_library", test_set=None, metric_threshold=0.8):
    """
    Train and validate LoRA adapters per primitive, reuse base model to save memory/time.
    """
    os.makedirs(library_dir, exist_ok=True)
    results = {}

    # Load base model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    base_model = prepare_model_for_int8_training(base_model)

    for primitive in new_primitives_to_train:
        primitive_name = primitive['name']
        print(f"\n[Phase 3] Training LoRA for primitive: {primitive_name}")

        satisfactory = False
        attempt = 0

        while not satisfactory:
            attempt += 1
            print(f"[Phase 3] Attempt #{attempt} for {primitive_name}")

            # Step 1: Generate Synthetic Data
            synthetic_data = generate_synthetic_data([primitive])

            # Step 2: Initialize new LoRA adapter (keep base model frozen)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            lora_model = get_peft_model(base_model, lora_config)

            # Step 2.1: Create Dataset and Trainer
            dataset = SyntheticPrimitiveDataset(synthetic_data, tokenizer)
            training_args = TrainingArguments(
                output_dir=f"{library_dir}/{primitive_name}",
                per_device_train_batch_size=4,
                num_train_epochs=3,
                logging_steps=10,
                save_steps=50,
                learning_rate=2e-4,
                fp16=True,
                remove_unused_columns=False
            )
            trainer = Trainer(
                model=lora_model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=training_args
            )

            # Step 2.2: Train LoRA adapter
            trainer.train()
            print(f"[Phase 3] LoRA training complete for {primitive_name}")

            # Step 3: Validate LoRA
            eval_results = evaluate_lora(lora_model, tokenizer, test_set)
            print(f"[Phase 3] Evaluation results: {eval_results}")

            # Step 4: Check metrics and save if satisfactory
            if any(value >= metric_threshold for value in eval_results.values()):
                satisfactory = True
                lora_model.save_pretrained(f"{library_dir}/{primitive_name}")
                print(f"[Phase 3] Saved LoRA adapter to {library_dir}/{primitive_name}")
            else:
                print(f"[Phase 3] Failed evaluation. Regenerating data and retraining...")

        results[primitive_name] = eval_results

    return results
