# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json , re
import uuid

from ..model_config import generate_text
from .utils import extract_json_from_text

from ..config import *
import numpy as np

# ---------------- Prompt Template for Phase 2 ----------------
system_prompt = f"""
You are an AI reasoning assistant that generates minimal programmatic primitives to solve a problem.

Rules:
1. Use existing primitives if they match the task. Otherwise, generate a new primitive.
2. Each primitive must include:
   - id: reuse the existing primitive's id if it exists; otherwise leave it as an empty string
   - name: short human-friendly name
   - description: one-sentence description
   - input: minimal input schema (field names/types)
   - output: minimal output schema (field names/types)
   - related_primitives: list of primitive IDs or names it often co-occurs with
   - status: 'existing' if reused, 'new' if generated

3. Produce a sequence in execution order.
4. Output must be valid JSON and contain ONLY the JSON array of primitives.
5. For new primitives, provide only the minimal info required to train later.

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

    old_primitives_text = ""
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
        old_primitives_text = f"\nExisting primitives:\n{json.dumps(summary, indent=2, ensure_ascii=False)}\n"

    user_prompt = f"Problem:\n{problem_text}\n"
    if domain_hint:
        user_prompt += f"Domain hint: {domain_hint}\n"
    user_prompt += old_primitives_text
    if analysis:
        user_prompt += f"\nProblem analysis:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}\n"

   # Add instructions for <start>/<end> markers
    user_prompt += '''\nGenerate only the sequence of primitives in execution order.
                Important:
                - Enclose the JSON array between <start> and <end>.
                - Output only valid JSON.
                - Do not include any extra text or code after the <end> marker.
                '''

    print("Calling LLM to generate primitive sequence...")
    raw_output = generate_text(model ,tokenizer, system_prompt, user_prompt,max_tokens=1500)
    # print("Raw LLM output for primitives:", raw_output)
    try:
        # json_text = extract_json_from_text(raw_output)
        primitives_sequence = parse_llm_json(raw_output)["primitives"]

    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from LLM output: {e}")

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

    retrieved = [primitive_id_map[i] for i in I[0]]

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











