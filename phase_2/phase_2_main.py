# phase_3_main.py

from .utils import *
from .generate_primitive import *

from ..config import *


def run_phase2(model, tokenizer, problem_text, analysis):
    """
    Phase 2: Reflective primitive planning pipeline.

    Steps:
      1. Retrieve relevant primitives from the library.
      2. Evaluate whether retrieved primitives are sufficient to solve the problem.
      3. If not sufficient, generate new primitives and produce a final ordered sequence.

    Args:
        model, tokenizer: LLM interface.
        problem_text (str): The original problem text.
        analysis (dict): Structured problem analysis (topics, tags, etc.)

    Returns:
        tuple: (final_sequence, new_primitives)
    """

    print("\n================= PHASE 2: REFLECTIVE PRIMITIVE PLANNING =================")

    # --------------------------------------------------------------
    # Step 1 — Retrieval
    # --------------------------------------------------------------
    print("\n[1] Retrieving relevant primitives...")
    retrieved_primitives = retrieve_primitives(analysis)

    if not retrieved_primitives:
        print("⚠️ No primitives retrieved from memory; starting from scratch.")
    else:
        print(f"✅ Retrieved {len(retrieved_primitives)} candidate primitives.")

    # --------------------------------------------------------------
    # Step 2 — Evaluate sufficiency of retrieved primitives
    # --------------------------------------------------------------
    print("\n[2] Evaluating sufficiency of retrieved primitives...")
    sufficiency_result = evaluate_primitive_sufficiency(
        model=model,
        tokenizer=tokenizer,
        problem_text=problem_text,
        analysis=analysis,
        retrieved=retrieved_primitives,
    )

    # reuse_ids = sufficiency_result.get("reuse", [])
    missing_caps = sufficiency_result.get("missing_capabilities", [])

    if not missing_caps:
        print(f"✅ Retrieved primitives appear sufficient.")
    else:
        print(f"🧠 Missing conceptual capabilities identified: {missing_caps}")

    # --------------------------------------------------------------
    # Step 3 — Generate final sequence (reuse + new primitives)
    # --------------------------------------------------------------
    print("\n[3] Generating reasoning sequence...")
    final_sequence = generate_primitives_with_reflection(
        model=model,
        tokenizer=tokenizer,
        problem_text=problem_text,
        analysis=analysis,
        retrieved=retrieved_primitives,
        sufficiency_result=sufficiency_result
    )

    # --------------------------------------------------------------
    # Step 4 — Separate and register new primitives
    # --------------------------------------------------------------
    new_prims = [p for p in final_sequence if p["status"] == "New"]

    if new_prims:
        print(f"Registering {len(new_prims)} new primitives into memory...")
        for p in new_prims:
            if "description" not in p:
                p["description"] = f"Auto-generated primitive: {p['name']}"
            try:
                add_primitive(p)
                print(f"[INFO] Added new primitive: {p['id']} — {p['name']}")
            except Exception as e:
                print(f"[ERROR] Could not add primitive {p['id']}: {e}")
    else:
        print("✅ No new primitives created.")

    # --------------------------------------------------------------
    # Debug summary
    # --------------------------------------------------------------
    print("\n────────────────────── Generated Reasoning Sequence ──────────────────────")
    for p in final_sequence:
        print(f"  Step {p.get('step', '?')}: {p['id']} ({p['status']}) — {p['name']}")
    print(f"─────────────────────────────────────────────────────────────────────────\n")
    print(f"Summary: {len(new_prims)} new / {len(final_sequence)} total primitives.\n")

    return final_sequence, new_prims


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
