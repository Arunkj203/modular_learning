
from synthetic_data import generate_synthetic_data_for_primitive
from lora_training import train_primitive, load_base_model , evaluate_lora

import os
import copy
import torch


# BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # Original Model

BASE_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # Test Model
OUTPUT_DIR = "./results/lora_adapters"

# OUTPUT_DIR = "./primitive_library"

def run_phase3(new_primitives_to_train, base_model_name=BASE_MODEL, metric_threshold=0.8):

    """
    Train and validate LoRA adapters per primitive.
    Save only adapters that pass the threshold.
    """
    # Load base model once
    base_model, tokenizer = load_base_model(base_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for primitive_name, dataset_path in new_primitives_to_train.items():
        print(f"\n=== Training primitive: {primitive_name} ===")

        # Load / generate dataset for this primitive
        dataset = generate_synthetic_data_for_primitive(dataset_path)
        test_set = dataset[:50]  # first 50 samples for testing

        # Create a fresh copy of the base model for LoRA training
        model_copy = copy.deepcopy(base_model)

        # Train LoRA adapter for this primitive
        trained_model = train_primitive(model_copy, tokenizer, dataset, primitive_name, OUTPUT_DIR)

        # Evaluate LoRA
        accuracy = evaluate_lora(trained_model, tokenizer, test_set, device)

        if accuracy >= metric_threshold:
            adapter_path = os.path.join(OUTPUT_DIR, primitive_name)
            trained_model.save_pretrained(adapter_path)
            print(f"✅ Saved adapter for {primitive_name} at {adapter_path}")
        else:
            print(f"❌ Adapter for {primitive_name} did not meet the threshold ({accuracy:.2f}). Skipping save.")
            return False
        
    return True

