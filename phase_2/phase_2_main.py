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
                "applied_on_state": f"{p.get('applied_on_state', '')}",
                "resulting_state": f"{p.get('resulting_state', '')}"
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


    # Step 3 - Separate new primitives (to train later)

    new_primitives_to_train = []
    primitive_sequence = []

    for p in sequence_primitives:
        name = p.get("name")
        status = p.get("status")

        if not name or not status:
            print(f"Skipping invalid primitive (missing name/status): {p}")
            continue

        existing_pid = name_to_id.get(name)  # canonical id if known
        if status == "Existing":
            if existing_pid:
                prim_entry = primitive_metadata.get(existing_pid)
                if prim_entry:
                    # Use the canonical primitive object (dict) for execution
                    primitive_sequence.append(prim_entry)
                else:
                    # name->id mapping exists but metadata missing: warn and skip
                    print(f"Warning: name_to_id mapped {name} -> {existing_pid} but metadata missing for that id. Skipping.")
            else:
                # LLM said 'Existing' but the name isn't in the library
                print(f"Warning: LLM marked primitive '{name}' as Existing but no canonical id found. Skipping for now.")
            continue

        # status == "New"
        if existing_pid:
            # LLM thinks it's new, but we already have a canonical primitive.
            # Use the existing primitive for execution (no retrain).
            prim_entry = primitive_metadata.get(existing_pid)
            if prim_entry:
                primitive_sequence.append(prim_entry)
            else:
                # weird case: mapping exists but metadata missing
                print(f"Warning: name_to_id mapped {name} -> {existing_pid} but metadata missing. Treating as new.")
                # fallthrough to create a new primitive below
        else:
            # Truly new primitive: create unique id, register minimal entry (but don't call add_primitive here)
            unique_suffix = uuid.uuid4().hex[:8]
            pid = f"{name}_{unique_suffix}"

            primitive_entry = {
                "id": pid,
                "name": name,
                "description": p.get("description", ""),
                "goal": p.get("goal", ""),
                "primitive_type": p.get("type", ""),
                "applied_on_state": p.get("applied_on_state", ""),
                "resulting_state": p.get("resulting_state", ""),
                "problem_type": p.get("problem_type", analysis.get("problem_type", "")),
                "methods": p.get("methods", analysis.get("selected_modules", [])),
                "tags": p.get("tags", analysis.get("tags", [])),
            }

            new_primitives_to_train.append(primitive_entry)
            primitive_sequence.append(pid)  # append id so later expansion can fetch full entry
            continue

        # If we reached here it means name->id existed but metadata missing and we fell through:
        # create new primitive entry as fallback
        if 'pid' not in locals() and existing_pid and not primitive_metadata.get(existing_pid):
            unique_suffix = uuid.uuid4().hex[:8]
            pid = f"{name}_{unique_suffix}"
            primitive_entry = {
                "id": pid,
                "name": name,
                "description": p.get("description", ""),
                "goal": p.get("goal", ""),
                "primitive_type": p.get("type", ""),
                "applied_on_state": p.get("applied_on_state", ""),
                "resulting_state": p.get("resulting_state", ""),
                "problem_type": p.get("problem_type", analysis.get("problem_type", "")),
                "methods": p.get("methods", analysis.get("selected_modules", [])),
                "tags": p.get("tags", analysis.get("tags", [])),
            }
            new_primitives_to_train.append(primitive_entry)
            primitive_sequence.append(pid)



    return primitive_sequence, new_primitives_to_train # new_count

