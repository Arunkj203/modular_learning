# Main code file
from .solve import *


from .model_config import get_model_and_tokenizer

from datasets import Dataset
from . import config as mem

def main():

    # dataset_name = sys.argv[1] # e.g., "SVAMP"

    # Load dataset
    dataset_name = "SVAMP"
    # dataset_name = "GSM8K"

    # Load model and tokenizer
    model , tokenizer = get_model_and_tokenizer()

    print(f"Model and tokenizer loaded for {dataset_name}.")

    # Testing 
    
    # solve(dataset_name,"train","Training", model, tokenizer)    

    #  # For generating phase analyses and executions - SVAMP

    # generate_phase1_analysis(dataset_name, "train", model, tokenizer)

# -------------------- Current Execution ------------------------------------------------------
    # Total - 10 batches of 70 each for SVAMP train set (700 problems)
    # Total batch generated till now - 0

    # for i in range(5, 11):
    #     generate_phase2_execution("SVAMP_train_phase1_analysis.json", model, tokenizer, batch_no=i, batch_size=70)
    
    for i in range(1,6):
        print(f"\nProcessing batch {i} with Batch Size - 70...")
        generate_phase3_execution(f"SVAMP_train_phase2_execution_batch{i}_{i*70}.json" , model, tokenizer)

    # # update bacth_no and batch_size as needed
# --------------------------------------------------------------------------


    # # For generating phase analyses and executions - GSM8K

    # generate_phase1_analysis("GSM8K", "train", model, tokenizer)
    # generate_phase2_execution("GSM8K_train_phase1_analysis.json", model, tokenizer, batch_no=1, batch_size=300)
    # generate_phase3_execution("GSM8K_train_phase2_execution.json", model, tokenizer)



if __name__ == "__main__":
    main()


'''
In gsm8k datastet,

# Load the GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main")

# Get the number of problems in train and test splits
train_count = len(dataset["train"])
test_count = len(dataset["test"])
Number of problems in train set: 7473 
Number of problems in test set: 1319
Total number of problems: 8792

'''

# Phase 2 - Starts at 7.55 PM
