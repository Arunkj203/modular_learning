
from .synthetic_data import generate_synthetic_data_for_primitive
from .lora_training import train_primitive , evaluate_lora

import os
import copy
import torch


from ..model_config import OUTPUT_DIR, DEVICE


def run_phase3(model, tokenizer ,new_primitives_to_train, metric_threshold=0.8):

    """
    Train and validate LoRA adapters per primitive.
    Save only adapters that pass the threshold.
    """
    # Load base model once
    base_model = model

    for primitive in new_primitives_to_train:
        name = primitive["name"]
        print(f"\n=== Training primitive: {name} ===")

        # Load / generate dataset for this primitive
        dataset = generate_synthetic_data_for_primitive(model ,tokenizer,primitive)
        test_set = dataset[:50]  # first 50 samples for testing

        # Create a fresh copy of the base model for LoRA training
        model_copy = copy.deepcopy(base_model)

        # Train LoRA adapter for this primitive
        trained_model = train_primitive(model_copy, tokenizer, dataset, primitive["id"])

        # Evaluate LoRA
        accuracy = evaluate_lora(trained_model, tokenizer, test_set)

        if accuracy >= metric_threshold:
            adapter_path = os.path.join(OUTPUT_DIR, primitive["id"])
            trained_model.save_pretrained(adapter_path)
            print(f"✅ Saved adapter for {name} at {adapter_path}")
        else:
            print(f"❌ Adapter for {name} did not meet the threshold ({accuracy:.2f}). Skipping save.")
            return False
        
    return True

