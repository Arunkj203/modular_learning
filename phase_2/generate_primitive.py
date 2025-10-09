# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json

from ..model_config import generate_text
import numpy as np

from ..config import *

# ---------------- Prompt Template for Phase 2 ----------------
system_prompt = '''
You are a reasoning model that decomposes a problem into a sequence of human-like cognitive primitives.

Definition:
A *primitive* is a minimal, reusable cognitive operation that transforms a problem state toward its solution. 
It represents a human-level skill — interpretable, composable, and generalizable.

Characteristics:
- Minimal cognitive transformation (atomic reasoning skill)
- Human-interpretable (clearly states the skill being applied)
- Reusable across problems (identified by ID and name)
- Composable into higher-level reasoning procedures
- One of four types:
  1. Perceptual — recognize or extract structure
  2. Transformational — modify a symbolic or numeric representation
  3. Control — decide sequence, subgoal, or next operation
  4. Evaluation — check progress, correctness, or termination

Rules:
- You are not solving the problem; only planning the reasoning process.
- Each step in your plan corresponds to one primitive.
- If a primitive is reused, list only `id`, `name`, and `"status": "Existing"`.
- If a new primitive is introduced, define all its fields with `"status": "New"`.
- Preserve logical step order.
- Output strictly follows the JSON schema below.

Output Schema:
<<START>>
{
  "primitive_sequence": [
    {
      // Existing primitive reuse
      "id": "<existing_primitive_id>",
      "name": "<existing_primitive_name>",
      "status": "Existing"
    },
    {
      // New primitive definition
      "id": "<new_primitive_id>",
      "name": "<new_primitive_name>",
      "description": "<short description of what this skill does>",
      "type": "<Perceptual | Transformational | Control | Evaluation>",
      "applied_on_state": "<inferred or hypothetical subgoal>",
      "resulting_state": "<expected next subgoal or transformation>",
      "status": "New"
    }
  ]
}
<<END>>

'''


def generate_primitives_from_problem(
    model, tokenizer,
    problem_text: str,
    summary: Optional[Dict[str, Any]] = None,
    analysis: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a sequence of primitives to solve the given problem.
    Reuses existing primitives by name, assigns new IDs to new ones.
    """



    user_prompt = f"""

            Problem to solve:
            {problem_text}

            Analysis context:
            {json.dumps(analysis, indent=2)}

            Available Primitives:
            {json.dumps(summary,indent=2)}

            Your Task:
            1. Decompose the given problem into a logical sequence of primitive applications needed to reach the solution.
            2. Reuse available primitives where possible (output only id, name, and status).
            3. If new primitives are required, define them completely with all metadata fields.
            4. Do not perform any calculations or produce final answers — only outline the reasoning plan.
            5. Return the response strictly following the provided JSON schema.

            """

    print("Calling LLM to generate primitive sequence...")

    complexity_estimate = len(tokenizer(system_prompt + user_prompt)['input_ids'])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate))

    primitives_sequence = generate_text(
        model, tokenizer, system_prompt, user_prompt,
        dynamic_max_tokens=dynamic_max_tokens
    )["primitive_sequence"]
    print(primitives_sequence)
    if not primitives_sequence:
        print("Raw op:",primitives_sequence)
        raise ValueError("LLM did not return any primitives.")
    

    # Ensure output is a list
    #if isinstance(primitives_sequence, dict):
    #    primitives_sequence = [primitives_sequence]


   
    return primitives_sequence


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

    query = f"{analysis['problem_type']} {' '.join(analysis['selected_modules'])} {' '.join(analysis['tags'])}"

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







