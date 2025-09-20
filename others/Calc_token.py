# pip install datasets tiktoken

from datasets import load_dataset
import tiktoken
import json, statistics
from generate_primitive import *


enc = tiktoken.get_encoding("cl100k_base")

# Load SVAMP dataset (train + test)
svamp_train = load_dataset("ChilleD/SVAMP", split="train")
svamp_test = load_dataset("ChilleD/SVAMP", split="test")
all_problems = [item["Body"] + " " + item["Question"] for item in svamp_train] + \
               [item["Body"] + " " + item["Question"] for item in svamp_test]

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# --- Stage C: Problem + Primitives to Generator ---
def solve_with_primitives(problem_text: str, primitive_library_path: str, provenance: Optional[str] = None):
    print("Analyzing problem...")
    analysis = analyze_problem(problem_text)
    print("Analysis:", analysis)

    print("Loading primitive library...")
    all_primitives = load_primitives(primitive_library_path)

    print("Filtering relevant primitives...")
    relevant = [p for p in all_primitives if any(tag in p.get("tags", []) for tag in analysis.get("tags", []))]
    print(f"Found {len(relevant)} relevant primitives.")

    domain_hint = analysis.get("domain", None)

    print("Generating primitives from problem...")
    new_primitives = generate_primitives_from_problem(
        problem_text,
        domain_hint=domain_hint,
        provenance=provenance or "pipeline_generated",
        old_primitives=relevant,
        analysis=analysis
    )

    if new_primitives:
        save_primitives(primitive_library_path, new_primitives)
