# phase_3/lora_training.py

import os
import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,SFTConfig

from modular_learning.model_config import get_model_and_tokenizer, OUTPUT_DIR, DEVICE



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


def train_primitive(copy_model, copy_tokenizer, dataset, primitive_id):
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
        output_dir=os.path.join(OUTPUT_DIR, primitive_id),
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
            model=copy_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            processing_class=copy_tokenizer,
            args=sft_config,
            peft_config=peft_config,
        )


    print(f"Training LoRA adapter for primitive '{primitive_id}'...")
    trainer.train()

    print("Training completed. Saving model...")
    return copy_model



# ------------------------
# Load LoRA-adapted model
# ------------------------
def load_lora_model(lora_path: str):

    # Load base model once
    base_model, tokenizer = get_model_and_tokenizer()
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model, tokenizer


# ------------------------
# Evaluate
# ------------------------

def evaluate_lora(model, tokenizer, test_data):
    model.eval()
    correct = 0

    for sample in test_data:
        # Construct the full prompt exactly like training
        prompt = sample["text"].rsplit("Output:", 1)[0] + "Output:"

        # Expected output (after "Output:")
        expected = sample["text"].rsplit("Output:", 1)[1].strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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






# def load_base_model(base_model_name: str):
#     """Load base model and tokenizer with quantization."""
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_name,
#         load_in_4bit=True,
#         device_map="auto"
#     )
#     model.config.use_cache = False
#     return model, tokenizer

