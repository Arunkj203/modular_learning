# phase_3/lora_training.py

import os
import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,SFTConfig

from ..model_config import get_model_and_tokenizer, OUTPUT_DIR, DEVICE



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
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected}")
            print(f"Generated: {generated}\n")

    accuracy = correct / len(test_data)
    print(f"Accuracy: {accuracy*100:.2f}%")
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

