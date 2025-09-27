
from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3
from .phase_4.phase_4_main import run_phase4

from .config import *
from datasets import load_dataset
from .utils import *

def solve(dataset_name,mode,mode_text, model, tokenizer):

    correct, total = 0, 0
    primitive_logs = []
    all_feedback = []  # Collect feedback for all problems

    dataset = load_dataset(dataset_path[dataset_name])


    load_memory()

    print(f"\n--- {mode_text} on {dataset_name} ---")

    for idx , problem in  enumerate(list(dataset[mode])[:20]):  # Limit to first 20 for testing
        print(f"\n=== Problem {idx+1} ===")

        '''  Phase 1: Problem Analysis'''

        processed, analysis = run_phase1(model, tokenizer , problem, dataset_name=dataset_name)
        gt = normalize_answer(processed["answer"])

        print("Phase 1 : Processed")

        '''  Phase 2: Primitive Generation  '''

        primitive_sequence , new_primitives_to_train = run_phase2(model, tokenizer ,processed["question"], analysis)

        print(f"Phase 2 : Primitive Sequence Generated")


        '''  Phase 3: Primitive Training and Testing  '''

        status = run_phase3(model, tokenizer ,new_primitives_to_train)
        if not status:
            print("Phase 3 failed. Exiting.")
            exit(1)

        print(f"Phase 3 completed. Trained {len(new_primitives_to_train)} new primitives.")
        # Note : Some changes need to made in phase 3 (In saving the lora adpaters , path changes etc)

        ''' Phase 4: Problem Solving + Feedback '''
        solution, steps, feedback_entries = run_phase4(model, tokenizer ,primitive_sequence, problem_text=processed["question"])

        print("Phase 4 : Problem Solved")

        # print("Steps:", steps)
        # print("Solution:", solution)

        # Collect all feedback
        all_feedback.extend(feedback_entries) 
        # Changes need to be made in phase 4 (return feedback entries)

        pred = normalize_answer(solution)

        if pred == gt:
                correct += 1
        total += 1


    
    acc = correct / total if total > 0 else 0

    return acc , feedback_entries



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

