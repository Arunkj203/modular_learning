# phase_3_main.py

from modular_learning.utils import *
from .generate_primitive import *






def run_phase2(model, tokenizer ,problem_text, analysis):

    """
    Phase 2: Generate a sequence of primitives to solve the problem.

    Args:
        problem_text (str): Original problem/question text
        analysis (dict): Analysis output from Phase 1

    Returns:
        dict: {
            "primitive_sequence": list of primitives (input/output/description),
            "new_primitives_to_train": list of new primitives (minimal schema)
        }
    """

    # Step 1 - Retrieve relevant primitives from library
    existing_primitives = retrieve_primitives(analysis)
    print(f"Retrieved {len(existing_primitives)} relevant primitives.")


    # Step 2 - Ask LLM to generate a sequence using old primitives or new ones
    sequence_primitives = generate_primitives_from_problem(
        model, tokenizer ,
        problem_text=problem_text,
        domain_hint=analysis.get("domain"),
        provenance="phase_1_pipeline",
        old_primitives=existing_primitives,
        analysis=analysis
    )
    print(f"LLM returned {len(sequence_primitives)} primitives in sequence.")


    # Step 3 - Separate new primitives (to train later)
    new_primitives_to_train = []
    primitive_sequence = []

    existing_ids = {p["id"] for p in existing_primitives}

    for p in sequence_primitives:
        # Minimal info for execution
        primitive_entry = {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "input": p.get("input_schema", {}),
            "output": p.get("output_schema", {}),
            "description": p.get("description", ""),
            "problem_type": p.get("problem_type", analysis.get("problem_type", "")),
            "domain": p.get("domain", analysis.get("domain", "")),
            "methods": p.get("methods", analysis.get("methods", [])),
            "tags": p.get("tags", analysis.get("tags", [])),
        }

        primitive_sequence.append(primitive_entry)


        # Collect new primitives that are not in existing library
        if primitive_entry["id"] not in existing_ids:
            new_primitives_to_train.append(primitive_entry)
            add_primitive(primitive_entry)


    return primitive_sequence , new_primitives_to_train
