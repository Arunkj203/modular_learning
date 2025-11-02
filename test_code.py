from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3

from .config import *
from .model_config import get_model_and_tokenizer
from datasets import load_dataset

# Load dataset
dataset_name = "SVAMP"
dataset = load_dataset(dataset_path[dataset_name])

problem = list(dataset["train"])[35]

# print("Problem:", problem)
# Load model and tokenizer
model, tokenizer = get_model_and_tokenizer()
 
print(f"Model and tokenizer loaded for {dataset_name}.")

print(f"\n--- Train on {dataset_name} ---")

# for idx , problem in  enumerate(list(dataset[mode])[:20]):  # Limit to first 20 for testing
print(f"\n=== Problem {1} ===")

'''  Phase 1: Problem Analysis'''
print(f"\nPhase 1 - Analysing...\n")

processed, analysis = run_phase1(model, tokenizer , problem, dataset_name=dataset_name)

#gt = normalize_answer(processed["answer"])

print("Phase 1 : Processed:\n",processed,"\nAnalysis:",analysis)



'''  Phase 2: Primitive Generation  

First retrieve relevant primitives from library
split into subtasks
the primitives to solve subtasks
Then generate a sequence of primitives to solve the problem
'''
print(f"\nPhase 2 - Primitive Sequence Generating...\n")

primitive_sequence , new_primitives_to_train = run_phase2(model, tokenizer ,processed["question"], analysis)

print(f"Phase 2 : Primitive Sequence Generated\n", primitive_sequence,"\nNew Primitives to train:", new_primitives_to_train)


'''  Phase 3: Primitive Training and Testing  '''
# This is trained , next step is to use this trained primitive in phase 4
# and see if it works correctly

# status = run_phase3(model, tokenizer ,new_primitives_to_train)
# if not status:
#    print("Phase 3 failed. Exiting.")
#    exit(1)

print(f"\nPhase 3 - Skipping...\n")

# print(f"Phase 3 completed. Trained {len(new_primitives_to_train)} new primitives.")
# Note : Some changes need to made in phase 3 (In saving the lora adpaters , path changes etc)

''' Phase 4: Problem Solving + Feedback 
with the sequence of primitives generated in phase 2,solve the problem
'''
print(f"\nPhase 4 - Solving...\n")

solution, steps, feedback_entries = run_phase3(model, tokenizer ,primitive_sequence, problem_text=processed["question"])

print("Phase 4 : Problem Solved")

print("Steps:", steps)
print("Solution:", solution)


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




# '''

# --- Train on SVAMP ---

# === Problem 1 ===

# Phase 1 - Analysing...


# Decomposing...

# Phase 1 : Processed:
#  {'id': 'chal-370', 'question': 'Edward spent $ 17. Then he received $ 10 from his friend. Now he has $ 7. How much did Edward have before he spent his money?', 'answer': '14', 'intermediate_steps': '( ( 17.0 - 10.0 ) + 7.0 )', 'type': 'Addition'}
# Analysis: 
# {'problem_type': 'word_problem', 
# 'domain': 'Arithmetic', 
# 'sub_domain': 'Money Word Problem', 
# 'tags': ['money', 'word_problem', 'arithmetic'], 
# 'topics': ['addition', 'subtraction'], 
# 'selected_reasoning_modules': ['Step-By-Step', 'Simplification'], 
# 'methods': ['addition', 'subtraction'], 
# 'decomposition_strategies': ['decompose_into_smaller_problems'], 
# 'decomposition_plan': [{'goal': 'find_the_amount_of_money_Edward_had_before_he_spent_it', 
# 'description': 'This is the core problem that we need to solve.'},
#  {'goal': 'find_the_amount_of_money_Edward_received_from_his_friend', 
#  'description': 'This is a supporting conceptual goal that we need to solve.'}], 
 
#  'subtasks': [{'step': 1, 'instruction': 'Identify the amount of money Edward spent.'}, 
#  {'step': 2, 'instruction': 'Identify the amount of money Edward received.'},
#    {'step': 3, 'instruction': 'Identify the amount of money Edward has.'}]}

# Phase 2 - Primitive Sequence Generating...

# Calling LLM to generate primitive sequence...
# Phase 2 : Primitive Sequence Generated
#  [{'id': 'find_the_amount_of_money_Edward_spent_05ce974f', 'name': 'find_the_amount_of_money_Edward_spent', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward spent.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_received_327604eb', 'name': 'find_the_amount_of_money_Edward_received', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward received.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_has_0899ccea', 'name': 'find_the_amount_of_money_Edward_has', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward has.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_had_before_he_spent_it_3837a93a', 'name': 'find_the_amount_of_money_Edward_had_before_he_spent_it', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward had before he spent it.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}]
# New Primitives to train: [{'id': 'find_the_amount_of_money_Edward_spent_05ce974f', 'name': 'find_the_amount_of_money_Edward_spent', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward spent.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_received_327604eb', 'name': 'find_the_amount_of_money_Edward_received', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward received.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_has_0899ccea', 'name': 'find_the_amount_of_money_Edward_has', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward has.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}, {'id': 'find_the_amount_of_money_Edward_had_before_he_spent_it_3837a93a', 'name': 'find_the_amount_of_money_Edward_had_before_he_spent_it', 'input': {}, 'output': {}, 'description': 'Find the amount of money Edward had before he spent it.', 'problem_type': 'word_problem', 'domain': 'Arithmetic', 'methods': ['addition', 'subtraction'], 'tags': ['money', 'word_problem', 'arithmetic']}]

# Phase 3 - Skipping...


# Phase 4 - Solving...


# '''
