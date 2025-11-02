# Main code file
from .solve import *


from .model_config import get_model_and_tokenizer

from datasets import Dataset
from . import config as mem

def main():

    # dataset_name = sys.argv[1] # e.g., "SVAMP"

    # Load dataset
    dataset_name = "SVAMP"

    # Load model and tokenizer
    model , tokenizer = get_model_and_tokenizer()

    print(f"Model and tokenizer loaded for {dataset_name}.")
<<<<<<< HEAD

    generate_phase2_execution("SVAMP_train_phase1_analysis.json", model, tokenizer)
    # load_memory()
=======
    
    solve(dataset_name,"train","Training", model, tokenizer)
    # generate_phase2_execution("SVAMP_train_phase1_analysis.json", model, tokenizer)
>>>>>>> 4994e4cc4c10ba209aff443dbd88d123c9787274

    print(f"\nTotal primitives in memory: {len(mem.primitive_metadata)}")
    for i, (pid, meta) in enumerate(mem.primitive_metadata.items()):
        print(f"{i+1:03d}. {pid}  →  {meta.get('name', '')}")
        if i >= 30:  # show only the first 30 to avoid flooding
            break


    # solve(dataset_name,"train","Training", model, tokenizer)
    # solve(dataset_name,,"Training", model, tokenizer)
    # generate_phase1_analysis(dataset_name, "train", model, tokenizer)

    # generate_phase4_execution("SVAMP_train_phase2_execution.json", model, tokenizer)

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

