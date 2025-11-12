# Main code file
from .solve import *


from .model_config import get_model_and_tokenizer

from . import config as mem





def main():

    # dataset_name = sys.argv[1] # e.g., "SVAMP"

    # Load dataset
    # dataset_name = "SVAMP"
    dataset_name = "GSM8K"

    # svamp = load_dataset("ChilleD/SVAMP", split="test")
    # print(f"Loaded SVAMP test: {len(svamp)} samples")

    gsm8k = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded GSM8k test: {len(gsm8k)} samples")

    # Load model and tokenizer
    model , tokenizer = get_model_and_tokenizer()

    print(f"Model and tokenizer loaded for {dataset_name}.")

    # Base_L tetsing:

    # SVAMP:
    # generate_phase1_analysis(svamp,"svamp", model, tokenizer, output_dir="Base_L")
    generate_phase2_execution("svamp_test_phase1_analysis.json", model, tokenizer, output_dir="Base_L")
    generate_phase3_execution("svamp_test_phase2_execution.json", model, tokenizer, output_dir="Base_L")

    # GSM8k:
    
    # generate_phase1_analysis(gsm8k,"gsm8k", model, tokenizer, output_dir="Base_L")
    # generate_phase2_execution("gsm8k_test_phase1_analysis.json", model, tokenizer, output_dir="Base_L")
    # generate_phase3_execution("gsm8k_test_phase2_execution.json", model, tokenizer, output_dir="Base_L")




    
    





    


if __name__ == "__main__":
    main()


'''

In SVAMP Dataset,

Test dataset len - 300;

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


# Testing 
    
    # solve(dataset_name,"train","Training", model, tokenizer)    

    #  # For generating phase analyses and executions - SVAMP

    # generate_phase1_analysis(dataset_name, "train", model, tokenizer)

# -------------------- Current Execution ------------------------------------------------------
    # Total - 10 batches of 70 each for SVAMP train set (700 problems)
    # Total batch generated till now - 0

    # for i in range(5, 11):
    #     generate_phase2_execution("SVAMP_train_phase1_analysis.json", model, tokenizer, batch_no=i, batch_size=70)
    
    # for i in range(1,6):
    #     print(f"\nProcessing batch {i} with Batch Size - 70...\n")
    #     generate_phase3_execution(f"SVAMP_train_phase2_execution_batch{i}_{i*70}.json" , model, tokenizer)

    # # update bacth_no and batch_size as needed
# --------------------------------------------------------------------------


    # For generating phase analyses and executions - GSM8K

    # print("\nGenerating Phase 1 Analysis for GSM8K Train Set...\n")
    # generate_phase1_analysis("GSM8K", "train", model, tokenizer)
    
    # for i in range(4,5):
    #     print(f"\nProcessing batch {i} with Batch Size - 200...")
    #     generate_phase2_execution(f"GSM8K_train_phase1_analysis_batch{i}_{i*200}.json", model, tokenizer)
    
    # for i in range(2,5):
    #     print(f"\nProcessing batch {i} with Batch Size - 200...\n")
    #     generate_phase3_execution(f"GSM8K_train_phase2_execution_batch{i}_{i*200}.json" , model, tokenizer)

