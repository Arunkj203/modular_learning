# phase_2/generate_primitive.py
# --------------------------------------------------------------
# Generates an abstract sequence of cognitive primitives (Phase 2)
# --------------------------------------------------------------

from typing import Dict, Any, List, Optional
import json , re , uuid

from ..model_config import generate_text
import numpy as np

from .. import config as mem


# ---------------------------------------------------------------------
# SYSTEM PROMPT: abstract planner, not solver
# ---------------------------------------------------------------------
system_prompt = """
You are a reasoning planner that constructs an abstract sequence of reusable cognitive primitives.

Goal:
- Produce a high-level reasoning plan (no numbers, names, or story content).
- Each primitive represents a general mental operation.
- You may reuse existing primitives listed in AVAILABLE_PRIMITIVES.
- If no suitable primitive exists, define ONE new abstract primitive.

Rules:
1. Re-use primitives whenever possible; copy IDs exactly as listed.
2. If a new one is required, assign an ID 'P_new###', keep it general, and mark "status": "New".
3. Do not mention concrete values or entities.
4. End with an Evaluation primitive.
5. Output valid JSON only in this schema:

{
  "primitive_sequence": [
    {"step": 1, "id": "P001", "name": "IDENTIFY_QUANTITIES", "status": "Existing"},
    {"step": 2, "id": "P002", "name": "IDENTIFY_OPERATION", "status": "Existing"},
    {"step": 3, "id": "P_new001", "name": "COMPARE_RATIOS", "status": "New"},
    {"step": 4, "id": "P004", "name": "EVALUATE_RESULT", "status": "Existing"}
  ]
}
<END_OF_SEQUENCE>
"""

MAX_PRIMITIVES = 12


# ---------------------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------------------
def generate_primitives_from_problem(
    model,
    tokenizer,
    problem_text: str,
    summary: Optional[List[Dict[str, Any]]] = None,
    analysis: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate an ordered sequence of primitive IDs/names describing
    a reasoning plan for the given problem type.
    """

    # --------------------------------------------------------------
    # Build concise user prompt
    # --------------------------------------------------------------
    existing_prims = [
        (p["id"], p["name"]) for p in (summary or [])
        if isinstance(p, dict) and "id" in p and "name" in p
    ]
    available_str = json.dumps(dict(existing_prims), indent=2)

    user_prompt = f"""
Problem Type: {analysis.get('problem_type', 'Unknown')}
Topics: {', '.join(analysis.get('topics', []))}
Modules: {', '.join(analysis.get('selected_modules', []))}

AVAILABLE_PRIMITIVES = {available_str}

Task:
Plan an ordered sequence (≤ {MAX_PRIMITIVES}) of primitives that can solve
this type of problem.  Do NOT include concrete values or story content.
"""

    print("Calling model to generate primitive plan...")

    # Estimate safe max tokens
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate))

    # --------------------------------------------------------------
    # Call model (ensure generate_text returns raw text)
    # --------------------------------------------------------------
    primitives_sequence = generate_text(
        model, tokenizer, system_prompt, user_prompt,
        dynamic_max_tokens=dynamic_max_tokens
    )["primitive_sequence"]

    if not primitives_sequence:
        print("Raw op:",primitives_sequence)
        raise ValueError("LLM did not return any primitives.")
    
    # --------------------------------------------------------------
    # Validate and clean up
    # --------------------------------------------------------------
    valid_ids = set(mem.primitive_metadata.keys())
    seen = set()
    clean_seq = []

    for p in primitives_sequence:
        pid = p.get("id")
        name = p.get("name", "").strip()

        if not pid or not name:
            continue

        # Deduplicate
        if pid in seen:
            continue
        seen.add(pid)

        # Check validity
        if pid in valid_ids:
            p["status"] = "Existing"
        else:
            # If model claims "Existing" but ID unknown -> fix
            if p.get("status") == "Existing":
                print(f"[WARN] Primitive {pid} not in ontology; relabeling as New.")
            p["id"] = f"P_new_{uuid.uuid4().hex[:5]}"
            p["status"] = "New"

        clean_seq.append(p)

    # --------------------------------------------------------------
    # Enforce at least one Evaluation primitive at end
    # --------------------------------------------------------------
    if not any("EVALUATE" in p["name"].upper() for p in clean_seq):
        eval_pid = (
            [pid for pid, prim in mem.primitive_metadata.items()
             if "EVALUATE" in prim["name"].upper()] or ["P004"]
        )[0]
        clean_seq.append({
            "step": len(clean_seq) + 1,
            "id": eval_pid,
            "name": "EVALUATE_RESULT",
            "status": "Existing"
        })

    print("Generated primitives:", [p["name"] for p in clean_seq])
    return clean_seq


# ---------------------------------------------------------------------
# Example retrieval helper (unchanged except for small fix)
# ---------------------------------------------------------------------
def retrieve_primitives(analysis, top_k=10, expand_related=True, depth=1, min_similarity=0.5):
    if mem.faiss_index is None or mem.faiss_index.ntotal == 0:
        print("No primitives in memory index yet.")
        return []

    query_parts = [
        analysis.get("problem_type", ""),
        " ".join(analysis.get("topics", [])),
        " ".join(analysis.get("selected_modules", [])),
        " ".join(analysis.get("tags", [])),
    ]
    query = " ".join(query_parts).strip()
    query_vec = mem.embed_model.encode(query).astype("float32")
    D, I = mem.faiss_index.search(np.array([query_vec]), top_k)

    retrieved = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        pid = mem.primitive_id_map.get(idx)
        if pid and pid in mem.primitive_metadata:
            sim_score = float(dist) if dist <= 1 else 1 / (1 + dist)
            if sim_score >= min_similarity:
                prim = mem.primitive_metadata[pid]
                prim["similarity_score"] = sim_score
                retrieved.append(prim)

    if expand_related and retrieved:
        expanded = set(p["id"] for p in retrieved)
        frontier = set(expanded)
        for _ in range(depth):
            next_frontier = set()
            for pid in frontier:
                next_frontier.update(mem.primitive_graph.neighbors(pid))
            expanded.update(next_frontier)
            frontier = next_frontier

        retrieved += [
            mem.primitive_metadata[pid]
            for pid in expanded
            if pid not in [p["id"] for p in retrieved]
        ]

    retrieved.sort(key=lambda x: -x.get("similarity_score", 0.0))
    return retrieved


# Storage for primitives and their embeddings


def add_primitive(primitive):
    """
    Add a primitive to both graph and FAISS vector index
    """


    pid = primitive["id"]
    mem.primitive_metadata[pid] = primitive

    # Add node to graph
    mem.primitive_graph.add_node(pid, **primitive)

    # Add edges for related primitives
    for related in primitive.get("related_primitives", []):
        mem.primitive_graph.add_edge(pid, related)

    # Build embedding using all relevant fields
    text = " ".join([
        primitive.get("name", ""),
        primitive.get("description", ""),
        primitive.get("domain", ""),
        primitive.get("problem_type", ""),
        " ".join(primitive.get("methods", [])),
        " ".join(primitive.get("tags", []))
    ])

    vec = mem.embed_model.encode(text).astype("float32")

    # Add to FAISS
    idx = mem.faiss_index.ntotal
    mem.faiss_index.add(np.array([vec]))
    mem.primitive_id_map[idx] = pid








