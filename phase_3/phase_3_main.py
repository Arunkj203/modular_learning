from .synthetic_data import generate_synthetic_data_for_primitive
from .lora_training import evaluate_lora

import os
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from ..model_config import OUTPUT_DIR


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
        dataset = generate_synthetic_data_for_primitive(base_model, tokenizer, primitive)
        test_set = dataset[:50]  # take first 50 for quick evaluation

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
        accuracy = evaluate_lora(trained_model, tokenizer, test_set)

        # --------------------------------------------------
        # 7. Save adapter only if it meets the threshold
        # --------------------------------------------------
        if accuracy >= metric_threshold:
            adapter_path = os.path.join(OUTPUT_DIR, primitive_id)
            trained_model.save_pretrained(adapter_path)  # saves only adapter weights
            print(f"Saved adapter for {name} at {adapter_path}")
        else:
            print(f"Adapter for {name} did not meet the threshold ({accuracy:.2f}). Skipping save.")
            return False

        # --------------------------------------------------
        # 8. Cleanup to free memory
        # --------------------------------------------------
        del trainer, trained_model
        torch.cuda.empty_cache()

    return True
