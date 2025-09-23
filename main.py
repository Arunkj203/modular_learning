# Main code file

from phase_1.phase_1_main import run_phase1
from phase_2.phase_2_main import run_phase2
from phase_3.phase_3_main import run_phase3
from phase_4.phase_4_main import run_phase4



from config import *
from datasets import load_dataset


# Load SVAMP dataset
dataset = load_dataset(DATASET_SVAMP)
dataset1 = load_dataset(DATASET_ASDIV)
dataset2= load_dataset(DATASET_GSM8K)
dataset3 = load_dataset(DATASET_MATH23K)


all_feedback = []  # Collect feedback for all problems


for problem in dataset['train']:

    '''  Phase 1: Problem Analysis'''

    processed, analysis = run_phase1(problem, dataset_name="SVAMP")


    '''  Phase 2: Primitive Generation  '''

    primitive_sequence , new_primitives_to_train = run_phase2(processed["question"], analysis)


    '''  Phase 3: Primitive Training and Testing  '''

    status = run_phase3(new_primitives_to_train)
    if not status:
        print("Phase 3 failed. Exiting.")
        exit(1)

    # Note : Some changes need to made in phase 3 (In saving the lora adpaters , path changes etc)

    ''' Phase 4: Problem Solving + Feedback '''
    solution, steps, feedback_entries = run_phase4(primitive_sequence, problem_text=processed["question"])

    print("Steps:", steps)
    print("Solution:", solution)

    # Collect all feedback
    all_feedback.extend(feedback_entries) 
    # Changes need to be made in phase 4 (return feedback entries)

# --- After all problems, save feedback dataset for batch training ---
import json
feedback_file = "./results/feedback_dataset.json"
with open(feedback_file, "w") as f:
    json.dump(all_feedback, f, indent=2)

# # --- Optional: retrain LoRA adapters in batch ---
# run_phase3(feedback_file)
# print("✅ LoRA adapters retrained with session feedback")