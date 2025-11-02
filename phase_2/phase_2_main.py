# phase_3_main.py

from .utils import *
from .generate_primitive import *

from ..config import *
import uuid


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
    # print(f"Retrieved {len(existing_primitives)} relevant primitives.")

    summary, name_to_id = [], {}
    if existing_primitives:
        for p in existing_primitives:
            summary.append({
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "goal": p.get("goal", ""),
                "type": p.get("primitive_type", ""),
            })
            name_to_id[p.get("name", "")] = p.get("id")


    # Step 2 - Ask LLM to generate a sequence using old primitives or new ones
    sequence_primitives = generate_primitives_from_problem(
        model, tokenizer ,
        problem_text=problem_text,
        summary=summary,
        analysis=analysis
    )
    # print(f"LLM returned {len(sequence_primitives)} primitives in sequence.")


    # --------------------------------------------------------------
    # Step 3 -  Separate new vs existing primitives
    # --------------------------------------------------------------
    new_prims = [p for p in sequence_primitives if p["status"] == "New"]

    # Optionally, register new primitives immediately in memory
    for p in new_prims:
        if "description" not in p:
            p["description"] = f"Auto-generated primitive: {p['name']}"
        try:
            add_primitive(p)   # safely add to memory / FAISS index
            print(f"[INFO] Added new primitive to memory: {p['id']} -> {p['name']}")
        except Exception as e:
            print(f"[ERROR] Failed to add primitive {p['id']}: {e}")

    # --------------------------------------------------------------
    # Debug summary
    # --------------------------------------------------------------
    print("\nGenerated Primitive Sequence:")
    for p in sequence_primitives:
        print(f"  Step {p.get('step', '?')}: {p['id']} ({p['status']}) - {p['name']}")

    print(f"\nSummary: {len(new_prims)} / {len(sequence_primitives)} new primitives.\n")

    return sequence_primitives, new_prims # new_count


# primitive_entry = {
#                 "id": pid,
#                 "name": name,
#                 "description": p.get("description", ""),
#                 "goal": p.get("goal", ""),
#                 "primitive_type": p.get("type", ""),
#                 "applied_on_state": p.get("applied_on_state", ""),
#                 "resulting_state": p.get("resulting_state", ""),
#                 "problem_type": p.get("problem_type", analysis.get("problem_type", "")),
#                 "methods": p.get("methods", analysis.get("selected_modules", [])),
#                 "tags": p.get("tags", analysis.get("tags", [])),
#             }
