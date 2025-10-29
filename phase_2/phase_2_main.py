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
    # new_count = 0


    for p in sequence_primitives:
            name = p.get("name")
            status = p.get("status")

            if not name or not status:
                print(f"Skipping invalid primitive (missing name/status): {p}")
                continue

            if status == "Existing":
                pid = name_to_id.get(name)
                if not pid:
                    # Fallback: treat as new if name not in known set
                    status = "New"

                else:
                    
                    prim_id = primitive_metadata.get(pid)
                    if not prim_id:
                        print(f"Warning: Primitive ID {pid} not found in metadata.")
                        continue
                    primitive_sequence.append(prim_id)
                    continue  # Move to next primitive in sequence


            if status == "New":
                unique_suffix = uuid.uuid4().hex[:8]
                pid = f"{name}_{unique_suffix}"

                # Minimal info for execution
                primitive_entry = {
                    "id": pid,
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "goal": p.get("goal", ""),
                    "primitive_type": p.get("type", ""),
                    "applied_on_state": p.get("applied_on_state", ""),
                    "resulting_state": p.get("resulting_state", ""),
                    "problem_type": p.get("problem_type", analysis.get("problem_type", "")),
                    "methods": p.get("methods", analysis.get("selected_modules", [])),
                    "tags": p.get("tags", analysis.get("tags", [])),
                }
                # add_primitive(primitive_entry)
                new_primitives_to_train.append(primitive_entry)
                primitive_sequence.append(pid)
                # new_count += 1



    return primitive_sequence, new_primitives_to_train # new_count

