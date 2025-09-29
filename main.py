# Main code file
from .solve import solve
import sys

from .model_config import get_model_and_tokenizer

from datasets import Dataset
import json

import warnings

warnings.simplefilter("ignore")      # Suppress Python warnings

def main():

    # dataset_name = sys.argv[1] # e.g., "SVAMP"

    # Load dataset
    dataset_name = "SVAMP"

    # Load model and tokenizer
    model , tokenizer = get_model_and_tokenizer()

    print(f"Model and tokenizer loaded for {dataset_name}.")

    train_acc , train_feedback_entries = solve(dataset_name,"train","Training", model, tokenizer)
    #test_acc , test_feedback_entries = solve(dataset_name,"test","Testing", model, tokenizer)

    

    # Save feedback entries for LoRA training
    #tokenized_datasets = prepare_lora_dataset_by_primitive(train_feedback_entries.extend(test_feedback_entries), tokenizer)

    # Save all primitive datasets together in one JSON file
    #save_file = "./lora_dataset_all.json"
    #with open(save_file, "w", encoding="utf-8") as f:
    #    json.dump(tokenized_datasets, f, ensure_ascii=False, indent=2)

    #print(f"Saved all primitive datasets to {save_file}")


    print(f"Train Accuracy on {dataset_name}: {train_acc*100:.2f}%")
    #print(f"Test Accuracy on {dataset_name}: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

def prepare_lora_dataset_by_primitive(feedback_entries, tokenizer, max_length=512):
    """
    Prepares tokenized dataset per primitive for LoRA training.
    
    Returns: dict { primitive_id: list_of_tokenized_entries }
    """
    # Group entries by primitive_id
    primitive_groups = {}
    for entry in feedback_entries:
        pid = entry["primitive_id"]
        if pid not in primitive_groups:
            primitive_groups[pid] = []
        # Choose corrected_output if invalid, else original output
        target_output = entry.get("corrected_output") if not entry.get("valid", True) else entry.get("output")
        primitive_groups[pid].append({
            "input": entry["input"],
            "output": target_output
        })
    
    # Tokenize datasets
    tokenized_datasets = {}
    for pid, entries in primitive_groups.items():
        dataset = Dataset.from_list(entries)

        def tokenize_fn(entry):
            prompt = f"Instruction: {entry['input']}\nResponse:"
            full_text = f"{prompt} {entry['output']}"
            tokenized = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
        # Convert to list of dicts for saving
        tokenized_datasets[pid] = tokenized_dataset[:]
    
    return tokenized_datasets

