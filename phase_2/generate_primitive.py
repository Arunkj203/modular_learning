# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json
from models_config.model_config import call_openrouter
from utils import extract_json_from_text

# ---------------- Prompt Template for Phase 2 ----------------
PRIMITIVE_SEQUENCE_PROMPT = """
You are an assistant that produces a sequence of **primitives** to solve a given problem.
Rules:
1. Use existing primitives if they match the task. Otherwise, generate a new primitive.
2. Each primitive must include:
   - id: short unique id (or reuse existing id)
   - name: short human-friendly name
   - description: one-sentence description
   - input: minimal input schema (field names/types)
   - output: minimal output schema (field names/types)
3. Produce a **sequence in execution order**.
4. Output must be valid JSON and contain **ONLY** the JSON array of primitives.
5. For new primitives, provide only the minimal info required to train later.
"""

def generate_primitives_from_problem(
    problem_text: str,
    domain_hint: Optional[str] = None,
    provenance: Optional[str] = None,
    old_primitives: Optional[List[Dict[str, Any]]] = None,
    analysis: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a sequence of primitives to solve the given problem.
    Existing primitives may be reused. New primitives are minimal.
    """

    old_primitives_text = ""
    if old_primitives:
        # Summarize existing primitives for LLM
        summary = []
        for p in old_primitives:
            summary.append({
                "id": p.get("id"),
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "input": p.get("input_schema", {}),
                "output": p.get("output_schema", {})
            })
        old_primitives_text = f"\nExisting primitives:\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n"

    analysis_text = ""
    if analysis:
        analysis_text = f"\nProblem analysis:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}\n"

    user_prompt = (
        f"{PRIMITIVE_SEQUENCE_PROMPT}\n\n"
        f"Problem:\n{problem_text}\n"
    )
    if domain_hint:
        user_prompt += f"\nDomain hint: {domain_hint}\n"
    user_prompt += old_primitives_text
    user_prompt += analysis_text
    user_prompt += "\nGenerate only the sequence of primitives in execution order."

    system_prompt = "You are an AI reasoning assistant that generates minimal programmatic primitives to solve a problem."

    print("Calling LLM to generate primitive sequence...")
    raw_output = call_openrouter(system_prompt, user_prompt)

    try:
        json_text = extract_json_from_text(raw_output)
        primitives_sequence = json.loads(json_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}\nLLM output:\n{raw_output}")

    if not isinstance(primitives_sequence, list):
        raise RuntimeError("LLM output JSON is not a list.")

    # Minimal validation
    valid_primitives = []
    for p in primitives_sequence:
        if not all(k in p for k in ["id", "description", "input", "output"]):
            print(f"Skipping invalid primitive: {p}")
            continue
        valid_primitives.append(p)

    return valid_primitives




























# # For primitive types

# #!/usr/bin/env python3
# """
# generate_primitives.py

# Given a natural-language problem description, call an LLM to generate
# a set of reusable primitives (unique, simple, with input/output schemas),
# validate and store them locally in an SQLite "primitive library".

# Supports OpenAI Chat completions (default). Includes a stub showing how
# to plug in a local LLaMA (llama-cpp-python) model if you prefer.

# Dependencies:
#   pip install openai sqlalchemy python-dotenv
# Optional (local model):
#   pip install llama-cpp-python

# Usage:
#   - Export OPENAI_API_KEY or set in a .env file
#   - python generate_primitives.py
# """

# import os , json , re , requests
# from typing import Dict, Any, List, Optional

# from models_config.model_config import call_openrouter
# from utils import *



# # ---------------- Code Generation Helper ----------------
# def primitive_to_stub(p: Dict[str, Any]) -> str:
#     """Generate a small python function stub for the primitive."""
#     in_fields = p["input_schema"].get("properties", {})
#     arg_list = []
#     for k, v in in_fields.items():
#         arg_list.append(k)
#     args = ", ".join(arg_list)
#     stub = f"def {p['id']}({args}):\n"
#     stub += f"    \"\"\"{p['description']}\n\n    Input schema: {json.dumps(p['input_schema'], ensure_ascii=False)}\n    Output schema: {json.dumps(p['output_schema'], ensure_ascii=False)}\n    \"\"\"\n"
#     stub += f"    # TODO: implement\n"
#     stub += f"    raise NotImplementedError('Implement primitive: {p['id']}')\n"
#     return stub


# # ---------------- Prompt Template ----------------
# PRIMITIVE_PROMPT_INSTRUCTIONS = """
# You are a reasoning engineer that extracts **simple, reusable primitives** from a problem description.
# A primitive is a minimal operation or function that can be composed to solve larger tasks.

# Requirements:
# 1. Return a JSON array called "primitives".
# 2. Each primitive must be an object with the following required fields:
#    - id: a short unique id string (lowercase, hyphenated)
#    - name: short human-friendly name
#    - description: one-sentence description of intent
#    - input_schema: JSON-schema-style description of inputs (fields and types)
#    - output_schema: JSON-schema-style description of outputs (fields and types)
#    - example_call: an example showing input and corresponding output (JSON objects)
#    - complexity: "O(1)", "O(n)", or "O(n log n)" etc. (estimate)
#    - reusable: boolean (True if likely reusable across tasks)
#    - tags: list of short tags
# 3. Optionally include:
#    - adapter_name: e.g. LoRA adapter name you expect to use when fine-tuning this primitive
#    - notes: any assumptions or edge cases
# 4. Ensure primitives are simple (each does one clear thing), and avoid overlap between primitives.
# 5. Output must be valid JSON and contain ONLY the JSON (no extra commentary).
# 6. If you include schema types, use simple types: string, integer, number, boolean, array, object.

# Produce at least 4 primitives, but avoid more than 20. Make sure fields are well-formed JSON.
# """





# # --- Primitive generator with context ---
# def generate_primitives_from_problem(
#     problem_text: str,
#     domain_hint: Optional[str] = None,
#     provenance: Optional[str] = None,
#     old_primitives: Optional[List[Dict[str, Any]]] = None,
#     analysis: Optional[Dict[str, Any]] = None
# ) -> List[Dict[str, Any]]:
    

#     old_primitives_text = ""
#     if old_primitives:
#         old_primitives_text = f"\nPreviously available primitives:\n{summarize_primitives(old_primitives)}\n"

#     analysis_text = ""
#     if analysis:
#         analysis_text = f"\nProblem analysis:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}\n"

#     user_prompt = (
#         f"{PRIMITIVE_PROMPT_INSTRUCTIONS}\n\n"
#         f"Problem:\n{problem_text}\n"
#     )
#     if domain_hint:
#         user_prompt += f"\nDomain hint: {domain_hint}\n"
#     user_prompt += old_primitives_text
#     user_prompt += analysis_text
#     user_prompt += "\nGenerate only new or adapted primitives that complement the old ones.\nReturn the JSON now."

#     system_prompt = (
#         "You are an assistant that extracts small reusable programmatic primitives "
#         "from problem descriptions. Reuse or adapt old primitives when possible, "
#         "and ensure new ones are unique and non-duplicative."
#     )

#     print("Calling LLM to extract primitives... (this may take a moment)")
#     raw = call_openrouter(system_prompt, user_prompt)

#     try:
#         json_text = extract_json_from_text(raw)
#         data = json.loads(json_text)
#     except Exception as e:
#         raise RuntimeError(f"Failed to extract JSON from LLM output: {e}\nLLM output:\n{raw}")

#     if not isinstance(data, (dict, list)):
#         raise RuntimeError("Parsed JSON is not an object or array.")

#     primitives = data.get("primitives") if isinstance(data, dict) and "primitives" in data else (data if isinstance(data, list) else None)
#     if primitives is None:
#         raise RuntimeError("JSON does not contain 'primitives' array or is not an array.")

#     saved = []
#     for p in primitives:
#         errors = validate_primitive_schema(p)
#         if errors:
#             print(f"Validation errors for primitive {p.get('id', p.get('name','<unknown>'))}: {errors}")
#             continue
#         p["provenance"] = provenance or "llm_generated"
#         saved.append(p)

#     return saved


