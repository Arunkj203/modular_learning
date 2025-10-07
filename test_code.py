from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3
from .phase_4.phase_4_main import run_phase4

from .config import *
from .model_config import get_model_and_tokenizer , generate_text
from datasets import load_dataset

# Load dataset
dataset_name = "SVAMP"
dataset = load_dataset(dataset_path[dataset_name])

problem = list(dataset["train"])[35].get("question_concat", "").strip()

print("Problem:", problem)
# Load model and tokenizer
model, tokenizer = get_model_and_tokenizer()
 
print(f"Model and tokenizer loaded for {dataset_name}.")


# ------------------------ PROMPT TEMPLATES ------------------------
system_prompt = '''
SYSTEM:
You are an expert mathematician and programmer. Follow these rules EXACTLY:
1) Output ONLY valid JSON and nothing else. Wrap it exactly between <start> and <end> markers with no extra text.
2) Follow this schema exactly:
{
  "problem": "<original problem text>",
  "sub_tasks": [{"task":"..."}],
  "solution_steps": [{"step":"..."}],
  "final_answer": <number or string>,
  "feasibility": "<explain if the numeric answer violates any implicit real-world constraints>",
  "assumptions": "<explicit assumptions used (e.g., overdraft allowed or not)>"
}
3) If the algebraic solution implies an intermediate negative value or other real-world impossibility, still return the algebraic solution in final_answer but state the feasibility and assumptions clearly.
4) Do not include any extra commentary, debugging text, or explanatory paragraphs outside the JSON.
5) Use exact arithmetic and show intermediate calculations in solution_steps.
'''



print("\nSVAMP Problem:\n")
user_prompt = f"Problem: {problem}"
result = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=600)
print(json.dumps(result, indent=4))




'''My idea '''

# # processed =  {'id': 'chal-777', 'question': "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?", 'answer': '145', 'intermediate_steps': '( 290.0 / 2.0 )', 'type': 'Common-Division'}
# # analysis = {'problem_type': 'algebra', 
# #             'domain': 'math', 'methods': ['isolation', 'simplification'], 
# #             'tags': ['linear equation'], 
# #             'subtasks': 
# #             [{'step': 1, 'instruction': 'Identify the variable to isolate'}, 
# #              {'step': 2, 'instruction': 'Move constants to the other side'}]
# #              }


# # primitive_sequence  = [{'id': 'Isolate_6acb9b6a', 'name': 'Isolate', 'input': {}, 'output': {}, 'description': 'Isolate a variable by moving all other terms to the other side of the equation', 'problem_type': 'algebra', 'domain': 'math', 'methods': ['isolation', 'simplification'], 'tags': ['linear equation']}, {'id': 'Simplify_1c05edee', 'name': 'Simplify', 'input': {}, 'output': {}, 'description': 'Simplify an equation by combining like terms', 'problem_type': 'algebra', 'domain': 'math', 'methods': ['isolation', 'simplification'], 'tags': ['linear equation']}]
# # new_primitives_to_train = [{'id': 'Isolate_6acb9b6a', 'name': 'Isolate', 'input': {}, 'output': {}, 'description': 'Isolate a variable by moving all other terms to the other side of the equation', 'problem_type': 'algebra', 'domain': 'math', 'methods': ['isolation', 'simplification'], 'tags': ['linear equation']}, {'id': 'Simplify_1c05edee', 'name': 'Simplify', 'input': {}, 'output': {}, 'description': 'Simplify an equation by combining like terms', 'problem_type': 'algebra', 'domain': 'math', 'methods': ['isolation', 'simplification'], 'tags': ['linear equation']}]

# print(f"\n--- Train on {dataset_name} ---")

# # for idx , problem in  enumerate(list(dataset[mode])[:20]):  # Limit to first 20 for testing
# print(f"\n=== Problem {1} ===")

# '''  Phase 1: Problem Analysis'''
# print(f"\nPhase 1 - Analysing...\n")

# processed, analysis = run_phase1(model, tokenizer , problem, dataset_name=dataset_name)

# #gt = normalize_answer(processed["answer"])

# print("Phase 1 : Processed:\n",processed,"\nAnalysis:",analysis)

# # analysis :
# # {'problem_type': 'combinations', 
# # 'domain': 'combinations', 
# # 'methods': ['intermediate'], 
# # 'tags': ['Combinations', 'Combinatorics']}




# '''  Phase 2: Primitive Generation  '''
# print(f"\nPhase 2 - Primitive Sequence Generating...\n")

# primitive_sequence , new_primitives_to_train = run_phase2(model, tokenizer ,processed["question"], analysis)

# print(f"Phase 2 : Primitive Sequence Generated\n", primitive_sequence,"\nNew Primitives to train:", new_primitives_to_train)

# #  Primitive Sequence Generated
# #  [{'id': 'combinations_66752315', 
# # 'name': 'combinations', 
# # 'input': {}, 
# # 'output': {}, 
# # 'description': 'Combinations', 
# # 'problem_type': 'combinations',
# #  'domain': 'combinations', 
# # 'methods': ['intermediate'], 
# # 'tags': ['Combinations', 'Combinatorics']}]


# # New Primitives to train: 
# # [{'id': 'combinations_66752315', 
# # 'name': 'combinations', 
# # 'input': {}, 
# # 'output': {}, 
# # 'description': 
# # 'Combinations', 
# # 'problem_type': 'combinations', 
# # 'domain': 'combinations', 
# # 'methods': ['intermediate'], 
# # 'tags': ['Combinations', 'Combinatorics']}]


# '''  Phase 3: Primitive Training and Testing  '''
# # This is trained , next step is to use this trained primitive in phase 4
# # and see if it works correctly

# # status = run_phase3(model, tokenizer ,new_primitives_to_train)
# # if not status:
# #    print("Phase 3 failed. Exiting.")
# #    exit(1)

# print(f"\nPhase 3 - Skipping...\n")

# # print(f"Phase 3 completed. Trained {len(new_primitives_to_train)} new primitives.")
# # Note : Some changes need to made in phase 3 (In saving the lora adpaters , path changes etc)

# ''' Phase 4: Problem Solving + Feedback '''
# print(f"\nPhase 4 - Solving...\n")

# solution, steps, feedback_entries = run_phase4(model, tokenizer ,primitive_sequence, problem_text=processed["question"])

# print("Phase 4 : Problem Solved")

# print("Steps:", steps)
# print("Solution:", solution)


# def normalize_answer(ans):
#     """Normalize numbers/strings for comparison."""
#     if ans is None:
#         return None
#     if isinstance(ans, str):
#         ans = ans.strip().lower()
#         # try to coerce to number if possible
#         try:
#             return str(float(ans))
#         except:
#             return ans
#     if isinstance(ans, (int, float)):
#         return str(float(ans))
#     return str(ans)




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
