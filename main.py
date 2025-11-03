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
    
    # solve(dataset_name,"train","Training", model, tokenizer)

    generate_phase1_analysis(dataset_name, "train", model, tokenizer)

    # generate_phase2_execution("SVAMP_train_phase1_analysis.json", model, tokenizer)
    # generate_phase3_execution("SVAMP_train_phase2_execution.json", model, tokenizer)



# print(f"\nTotal primitives in memory: {len(mem.primitive_metadata)}")
# for i, (pid, meta) in enumerate(mem.primitive_metadata.items()):
#     print(f"{i+1:03d}. {pid}  →  {meta.get('name', '')}")
#     # if i >= 30:  # show only the first 30 to avoid flooding
#     #     break

if __name__ == "__main__":
    main()

