# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json
import uuid

from ..model_config import generate_text

from ..config import *
import numpy as np

# ---------------- Prompt Template for Phase 2 ----------------
system_prompt = f"""
You are an AI reasoning assistant that generates minimal programmatic primitives to solve a problem.

Rules:
1. Use existing primitives if they match a subtask. Otherwise, generate a new primitive.
2. Each primitive must include:
   - id: reuse existing primitive's id if applicable; otherwise leave empty
   - name: short human-friendly name
   - description: one-sentence description
   - input: minimal input schema (field names/types)
   - output: minimal output schema (field names/types)
   - related_primitives: list of primitive IDs or names it often co-occurs with
   - status: 'existing' if reused, 'new' if generated
3. Generate primitives in **execution order**, respecting subtask dependencies.
4. For new primitives, provide only minimal info required for later LoRA training.
5. Output must be **valid JSON**:
   - The outermost structure MUST be a JSON array `[...]`.
   - Each element MUST be a full JSON object `{...}`.
   - Objects MUST be separated by commas.
   - Do not include comments, trailing commas, or extra keys.
6. Wrap the array strictly between `<start>` and `<end>` markers.
"""


def generate_primitives_from_problem(
    model ,tokenizer,  
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

    summary = []
    existing_ids = []

    if old_primitives:
        # Summarize existing primitives for LLM
        for p in old_primitives:
            summary.append({
                # "id": p.get("id"),
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "input": p.get("input_schema", {}),
                "output": p.get("output_schema", {})
            })

            existing_ids.append(p.get("id"))

    
    # ---------------- Split analysis fields ----------------
    if analysis:
        # Copy analysis to avoid modifying original
        analysis_copy = analysis.copy()
        
        # Pop subtasks from analysis
        subtasks = analysis_copy.pop("subtasks", [])
        domain = analysis_copy.pop("domain","")
    else:
        analysis_copy = {}
        subtasks = []
        domain = ""


    # ---------------- Prepare subtasks ----------------
    subtasks_text = ""
    if subtasks:
        subtasks_text = json.dumps(subtasks, indent=2, ensure_ascii=False)


    user_prompt = f"""
            Problem:
            {problem_text}

            Domain hint: {domain_hint or 'None'} - {domain}

            Existing primitives (if any):
            {json.dumps(summary, indent=2, ensure_ascii=False)}

            Problem analysis:
            {json.dumps(analysis_copy, indent=2, ensure_ascii=False)}

            Subtasks:
            {subtasks_text}


            Instructions for LLM:
            - The 'subtasks' from analysis represent the logical steps to solve the problem.
            - Map each subtask to one or more primitives, reusing existing ones if possible.
            - Each primitive should be atomic, minimal, and solvable independently.
            - Output only the JSON array of primitives, between <start> and <end>.
            - Do NOT include any extra text outside the markers.
            """
    

    print("Calling LLM to generate primitive sequence...")

    last_error = None
    error = False
    for attempt in range(1, Retries + 1):
        raw = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=2500)
        try:
            # json_text = extract_json_from_text(raw_output)
            primitives_sequence = parse_raw_op_with_markers(raw)
            error = False
            break
        except Exception as e:
            last_error = e
            error = True
            print(f"[WARN] Attempt {attempt} failed: {e}")
            # optional: short delay before retry
    
    if error:
        # If all attempts failed, raise
        raise RuntimeError(
            f"Could not parse JSON after {Retries} attempts. "
            f"Last error: {last_error}\nLast LLM output:\n{raw}"
        )


    # Ensure it's a list of primitives
    if isinstance(primitives_sequence, dict):
        # wrap single object in a list
        primitives_sequence = [primitives_sequence]

    # ---------------- Post-process to assign IDs ----------------
    for p in primitives_sequence:
        # Reuse existing ID if LLM says 'existing' and name matches
        if p.get("status") == "existing" and p.get("id") in existing_ids:
            pass  # keep the existing ID
        elif p.get("status") == "new" or (p.get("status") == "existing" and p.get("id") not in existing_ids):            # Generate a new unique ID for new primitives or if LLM is unsure
            unique_suffix = uuid.uuid4().hex[:8]
            p['id'] = f"{p['name']}_{unique_suffix}"
        else:
            raise RuntimeError(f"Failed to assign ID for primitive: {p}")

    
    # Minimal validation
    valid_primitives = []
    for p in primitives_sequence:
        if not all(k in p for k in ["id", "description", "input", "output"]):
            print(f"Skipping invalid primitive: {p}")
            continue
        valid_primitives.append(p)

    return valid_primitives

# Storage for primitives and their embeddings



def add_primitive(primitive):
    """
    Add a primitive to both graph and FAISS vector index
    """
    pid = primitive["id"]
    primitive_metadata[pid] = primitive

    # Add node to graph
    primitive_graph.add_node(pid, **primitive)

    # Add edges for related primitives
    for related in primitive.get("related_primitives", []):
        primitive_graph.add_edge(pid, related)

    # Build embedding using all relevant fields
    text = " ".join([
        primitive.get("name", ""),
        primitive.get("description", ""),
        primitive.get("domain", ""),
        primitive.get("problem_type", ""),
        " ".join(primitive.get("methods", [])),
        " ".join(primitive.get("tags", []))
    ])

    vec = embed_model.encode(text).astype("float32")

    # Add to FAISS
    idx = faiss_index.ntotal
    faiss_index.add(np.array([vec]))
    primitive_id_map[idx] = pid


def retrieve_primitives(analysis, top_k=10, expand_related=True, depth=1):
    """
    Retrieve primitives based on query and optionally expand via related primitives graph
    """

    query = f"{analysis['problem_type']} {analysis['domain']} {' '.join(analysis['methods'])} {' '.join(analysis['tags'])}"

    # Semantic search
    query_vec = embed_model.encode(query).astype("float32")
    D, I = faiss_index.search(np.array([query_vec]), top_k)

    if primitive_id_map is None or len(primitive_id_map) == 0:
        return []

    retrieved = [primitive_id_map[i] for i in I[0] if i != -1]

    # Expand using graph relationships
    if expand_related:
        expanded = set(retrieved)
        frontier = set(retrieved)
        for _ in range(depth):
            next_frontier = set()
            for pid in frontier:
                next_frontier.update(primitive_graph.neighbors(pid))
            expanded.update(next_frontier)
            frontier = next_frontier
        return [primitive_metadata[pid] for pid in expanded]

    return [primitive_metadata[pid] for pid in retrieved]









