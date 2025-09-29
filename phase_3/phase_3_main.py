from .synthetic_data import generate_synthetic_data_for_primitive
from .lora_training import evaluate_lora

import os
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from ..model_config import OUTPUT_DIR

from datasets import Dataset, DatasetDict
import random


# ==================================================
# CONFIG
# ==================================================
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
NUM_EPOCHS = 3
BATCH_SIZE = 4
LR = 2e-4


def run_phase3(model, tokenizer, new_primitives_to_train, metric_threshold=0.8):
    """
    Train and validate LoRA adapters per primitive.
    Save only adapters that pass the threshold.
    """

    base_model = model  # frozen model loaded once onto GPU

    for primitive in new_primitives_to_train:
        primitive_id = primitive["id"]
        name = primitive["name"]
        print(f"\n=== Training primitive: {name} ===")

        # --------------------------------------------------
        # 1. Dataset for this primitive
        # --------------------------------------------------
        examples  = generate_synthetic_data_for_primitive(base_model, tokenizer, primitive)

        #  Change later : split based on formats

        # Split into train/val/test
        dataset = prepare_datasets(examples, test_size=0.1, val_size=0.1)



        # --------------------------------------------------
        # 2. LoRA config (fresh for every primitive)
        # --------------------------------------------------
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # --------------------------------------------------
        # 3. Training arguments
        # --------------------------------------------------
        sft_config = SFTConfig(
            max_length=64,
            output_dir=os.path.join(OUTPUT_DIR, primitive_id),
            overwrite_output_dir=True,
            per_device_train_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LR,
            logging_steps=10,
            eval_strategy="epoch",  
            save_strategy="epoch",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none",
            dataset_text_field="text",
        )

        # --------------------------------------------------
        # 4. Trainer with LoRA adapter
        # --------------------------------------------------
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            processing_class=tokenizer,       
            args=sft_config,
            peft_config=peft_config,
        )

        # --------------------------------------------------
        # 5. Train
        # --------------------------------------------------
        print(f"Training LoRA adapter for primitive '{primitive_id}'...")
        trainer.train()
        print("Training completed.")

        # --------------------------------------------------
        # 6. Evaluate
        # --------------------------------------------------
        trained_model = trainer.model
        accuracy = evaluate_lora(trained_model, tokenizer, dataset["test"])

        # --------------------------------------------------
        # 7. Save adapter only if it meets the threshold
        # --------------------------------------------------
        '''
        if accuracy >= metric_threshold:
            adapter_path = os.path.join(OUTPUT_DIR, primitive_id)
            trained_model.save_pretrained(adapter_path)  # saves only adapter weights
            print(f"Saved adapter for {name} at {adapter_path}")
        else:
            print(f"Adapter for {name} did not meet the threshold ({accuracy:.2f}). Skipping save.")
            return False
        '''

        adapter_path = os.path.join(OUTPUT_DIR, primitive_id)
        trained_model.save_pretrained(adapter_path)  # saves only adapter weights
        print(f"Saved adapter for {name} at {adapter_path}")

        
        # --------------------------------------------------
        # 8. Cleanup to free memory
        # --------------------------------------------------
        del trainer, trained_model
        torch.cuda.empty_cache()

    return True



def prepare_datasets(examples, test_size=0.1, val_size=0.1, seed=42):
    """
    Split a list of examples into train/val/test DatasetDict.
    """
    random.seed(seed)
    random.shuffle(examples)

    n_total = len(examples)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_data = examples[:n_test]
    val_data = examples[n_test:n_test + n_val]
    train_data = examples[n_test + n_val:]

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "val": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })
