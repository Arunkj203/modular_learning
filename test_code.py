from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3
from .phase_4.phase_4_main import run_phase4

from .config import *
from .model_config import get_model_and_tokenizer

# Load dataset
dataset_name = "SVAMP"

problem = {'ID': 'chal-777', 
           'Body': "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups", 
           'Question': 'How big is each group of bananas?', 
           'Equation': '( 290.0 / 2.0 )', 
           'Answer': '145', 
           'Type': 'Common-Division', 
           'question_concat': "There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups How big is each group of bananas?"
           }

# Load model and tokenizer
model, tokenizer = get_model_and_tokenizer()

print(f"Model and tokenizer loaded for {dataset_name}.")

print(f"\n--- Train on {dataset_name} ---")

# for idx , problem in  enumerate(list(dataset[mode])[:20]):  # Limit to first 20 for testing
print(f"\n=== Problem {1} ===")

'''  Phase 1: Problem Analysis'''

processed, analysis = run_phase1(model, tokenizer , problem, dataset_name=dataset_name)
#gt = normalize_answer(processed["answer"])

print("Phase 1 : Processed:\n",processed,"\nAnalysis:",analysis)

# analysis :
# {'problem_type': 'combinations', 
# 'domain': 'combinations', 
# 'methods': ['intermediate'], 
# 'tags': ['Combinations', 'Combinatorics']}




'''  Phase 2: Primitive Generation  '''

primitive_sequence , new_primitives_to_train = run_phase2(model, tokenizer ,processed["question"], analysis)

print(f"Phase 2 : Primitive Sequence Generated\n", primitive_sequence,"\nNew Primitives to train:", new_primitives_to_train)

#  Primitive Sequence Generated
#  [{'id': 'combinations_66752315', 
# 'name': 'combinations', 
# 'input': {}, 
# 'output': {}, 
# 'description': 'Combinations', 
# 'problem_type': 'combinations',
#  'domain': 'combinations', 
# 'methods': ['intermediate'], 
# 'tags': ['Combinations', 'Combinatorics']}]


# New Primitives to train: 
# [{'id': 'combinations_66752315', 
# 'name': 'combinations', 
# 'input': {}, 
# 'output': {}, 
# 'description': 
# 'Combinations', 
# 'problem_type': 'combinations', 
# 'domain': 'combinations', 
# 'methods': ['intermediate'], 
# 'tags': ['Combinations', 'Combinatorics']}]


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
