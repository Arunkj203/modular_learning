# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json
import uuid

from ..model_config import generate_text

from ..config import *
import numpy as np

# ---------------- Prompt Template for Phase 2 ----------------
system_prompt = f"""
You are a reasoning assistant that maps problem subtasks to minimal programmatic primitives.

Rules:
1. Reuse existing primitives if they fit the subtask goal.
2. If no existing primitive fits, create a new one with:
   - name
   - description
   - goal (summarized subtask)
   - status: "new"
3. Always output in execution order.
4. For reused primitives, include:
   - id
   - name
   - status: "existing"
5. The entire output must be valid JSON between <start> and <end>.
6. The outer structure must be a JSON array `[...]`.
7. No explanations, no text outside markers.
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
        subtasks_text = json.dumps(subtasks,separators=(",", ":"), ensure_ascii=False)
    
    user_prompt = f"""
                Problem:
                {problem_text}

                Problem analysis:
                    {json.dumps(analysis_copy,separators=(",", ":"), ensure_ascii=False)}

                Subtasks to solve:
                {json.dumps(analysis['subtasks'], separators=(',', ':'), ensure_ascii=False)}

                Relevant existing primitives:
                {json.dumps(summary, separators=(',', ':'), ensure_ascii=False)}

                Instructions:
                - For each subtask, try to find the best existing primitive.
                - If one fits, include its id and name with status 'existing'.
                - If none fit, create a minimal new primitive (include goal and description).
                - Output strictly valid JSON array between <start> and <end>.
                """

    print("Calling LLM to generate primitive sequence...")

    last_error = None
    error = False

    # Calculate dynamic max_tokens based on complexity
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)['input_ids'])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate )) 


    primitives_sequence = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=dynamic_max_tokens)
       

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









