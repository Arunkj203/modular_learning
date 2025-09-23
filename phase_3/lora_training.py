# phase_3/lora_training.py

import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,SFTConfig



# ==================================================
# CONFIG
# ==================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
NUM_EPOCHS = 3
BATCH_SIZE = 4
LR = 2e-4


# ==================================================
# HELPERS
# ==================================================
def load_base_model(base_model_name: str):
    """Load base model and tokenizer with quantization."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map="auto"
    )
    model.config.use_cache = False
    return model, tokenizer



def train_primitive(model, tokenizer, dataset, primitive_name , OUTPUT_DIR):
    """Train one primitive as a LoRA adapter."""

    # LoRA configuration


    # fresh LoRA config for this primitive
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training arguments : Create SFTConfig
    
    sft_config = SFTConfig(
        max_length=64,
        output_dir=os.path.join(OUTPUT_DIR, primitive_name),
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        eval_strategy ="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataset_text_field="text",

    )

    
    # SFT Trainer setup

    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            processing_class=tokenizer,
            args=sft_config,
            peft_config=peft_config,
        )


    print(f"Training LoRA adapter for primitive '{primitive_name}'...")
    trainer.train()

    print("Training completed. Saving model...")
    return model



# ------------------------
# Load LoRA-adapted model
# ------------------------
def load_lora_model(base_model_name: str, lora_path: str):
    base_model, tokenizer = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer


# ------------------------
# Evaluate
# ------------------------

def evaluate_lora(model, tokenizer, test_data, device):
    model.eval()
    correct = 0

    for sample in test_data:
        # Construct the full prompt exactly like training
        prompt = sample["text"].rsplit("Output:", 1)[0] + "Output:"

        # Expected output (after "Output:")
        expected = sample["text"].rsplit("Output:", 1)[1].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=64,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Check if expected output is in generated text
        if expected in generated:
            correct += 1
        else:
            print(f"❌ Prompt: {prompt}")
            print(f"Expected: {expected}")
            print(f"Generated: {generated}\n")

    accuracy = correct / len(test_data)
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    return accuracy




