
from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3
from .phase_4.phase_4_main import run_phase4

from .config import *
from datasets import load_dataset

import os


max_errors = 10
errors = 0 

def solve(dataset_name, mode, mode_text, model, tokenizer, log_dir="logs"):

    correct, total = 0, 0
    primitive_logs = []
    all_feedback = []  # Collect feedback for all problems

    use_lora = False
    dataset = load_dataset(dataset_path[dataset_name])
    load_memory()

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset_name}_{mode}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== {mode_text} on {dataset_name} ===\n\n")

        for idx, problem in enumerate(list(dataset[mode])[:5]):  # limit for testing
            print(f"=== Problem {idx+1} ===")

            f.write(f"\n=== Problem {idx+1} ===\n")
            
            try:

                # Phase 1: Problem Analysis
                processed, analysis = run_phase1(model, tokenizer, problem, dataset_name=dataset_name)
                gt = normalize_answer(processed["answer"])
                f.write(f"\nQuestion:\n{processed['question']}\n")
                f.write(f"\nGround Truth Answer:\n{gt}\n")
                f.write(f"\nPhase 1 - Analysis:\n{analysis}\n")
                print(f"Phase 1 - Analysed")
                

                # Phase 2: Primitive Generation
                primitive_sequence, new_primitives_to_train = run_phase2(model, tokenizer, processed["question"], analysis)
                f.write("\nPhase 2 - Primitive Sequence:\n")
                for prim in primitive_sequence:
                    f.write(f"  ID: {prim['id']}, Name: {prim.get('name','')}, Desc: {prim.get('description','')}\n")

                print(f"\nPhase 2 - Primitive Sequence Generated")
                
                # Optional Phase 3: Training
                if use_lora:
                    status = run_phase3(model, tokenizer, new_primitives_to_train)
                    if not status:
                        f.write("\nPhase 3 failed. Exiting.\n")
                        exit(1)
                    f.write(f"\nPhase 3 completed. Trained {len(new_primitives_to_train)} new primitives.\n")
                    # Note : Some changes need to made in phase 3 (In saving the lora adpaters , path changes etc)

                # Phase 4: Problem Solving
                solution, steps, feedback_entries = run_phase4(
                    model, tokenizer, primitive_sequence, problem_text=processed["question"]
                )

                f.write("\nPhase 4 - Execution Steps:\n")
                for step in steps:
                    f.write(f"  Primitive {step['primitive_id']} ({step['name']}):\n")
                    f.write(f"    Input: {step['input']}\n")
                    f.write(f"    Output: {step['output']}\n")

                f.write(f"\nFinal Solution:\n{solution}\n")
                
                print(f"====== Problem {idx+1} Spolved =======")

                # Collect all feedback
                # all_feedback.extend(feedback_entries) 
                # Changes need to be made in phase 4 (return feedback entries)


                # Track accuracy
                pred = normalize_answer(solution)
                if pred == gt:
                    correct += 1
                total += 1

            except Exception as e:
                errors += 1
                f.write(f"\n[ERROR] Problem {idx+1} failed: {e}\n")
                if errors >= max_errors:
                    f.write(f"\n[ABORT] Too many errors ({errors}). Stopping early.\n")
                    break

        # Write accuracy at the end
        acc = correct / total if total > 0 else 0
        f.write(f"\n\n=== Accuracy: {acc:.2f} ({correct}/{total}) ===\n")

    return acc, feedback_entries



def normalize_answer(ans):
    """Normalize numbers/strings for comparison."""
    if ans is None:
        return None
    if isinstance(ans, str):
        ans = ans.strip().lower()
        # try to coerce to number if possible
        try:
            return str(float(ans))
        except:
            return ans
    if isinstance(ans, (int, float)):
        return str(float(ans))
    return str(ans)



