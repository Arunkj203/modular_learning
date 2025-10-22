# phase_2/generate_primitive.py

from typing import Dict, Any, List, Optional
import json

from ..model_config import generate_text
import numpy as np

from ..config import *

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

-----------------------------------------
Few-shot Examples:
-----------------------------------------

Example 1:
Problem: "John has 3 apples and buys 2 more. How many apples does he have?"
Available Primitives: []
Analysis context: Basic addition of quantities.

<<START>>
{
  "primitive_sequence": [
    {
      "id": "P001",
      "name": "Identify Quantities",
      "description": "Recognize all numeric quantities in the problem.",
      "type": "Perceptual",
      "applied_on_state": "Raw problem text",
      "resulting_state": "List of quantities: [3, 2]",
      "status": "New"
    },
    {
      "id": "P002",
      "name": "Identify Operation",
      "description": "Determine which mathematical operation connects the quantities.",
      "type": "Perceptual",
      "applied_on_state": "Recognized quantities",
      "resulting_state": "Operation: addition",
      "status": "New"
    },
    {
      "id": "P003",
      "name": "Combine Quantities",
      "description": "Apply the identified operation on quantities.",
      "type": "Transformational",
      "applied_on_state": "Operation: addition, Quantities: [3, 2]",
      "resulting_state": "Intermediate result (5 apples)",
      "status": "New"
    },
    {
      "id": "P004",
      "name": "Evaluate Result",
      "description": "Check that the final result answers the question.",
      "type": "Evaluation",
      "applied_on_state": "Intermediate result",
      "resulting_state": "Final validated answer",
      "status": "New"
    }
  ]
}
<<END>>

Example 2:
Problem: "A train travels 60 km in 2 hours. What is its speed?"
Available Primitives: [P001, P002, P003, P004]
Analysis context: Division for rate calculation.

<<START>>
{
  "primitive_sequence": [
    {
      "id": "P001",
      "name": "Identify Quantities",
      "status": "Existing"
    },
    {
      "id": "P002",
      "name": "Identify Operation",
      "status": "Existing"
    },
    {
      "id": "P005",
      "name": "Recognize Relationship Type",
      "description": "Determine whether the relation involves addition, subtraction, multiplication, or division.",
      "type": "Perceptual",
      "applied_on_state": "Recognized quantities and context",
      "resulting_state": "Identified relationship: division",
      "status": "New"
    },
    {
      "id": "P006",
      "name": "Apply Division Operation",
      "description": "Compute rate as distance divided by time.",
      "type": "Transformational",
      "applied_on_state": "Distance = 60, Time = 2",
      "resulting_state": "Speed = 30 km/h",
      "status": "New"
    },
    {
      "id": "P004",
      "name": "Evaluate Result",
      "status": "Existing"
    }
  ]
}
<<END>>

-----------------------------------------
Output Schema (for your problem):
-----------------------------------------
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

MAX_PRIMITIVES = 12



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
        {json.dumps(analysis or {}, indent=2)}

        Available Primitives summary:
        {json.dumps(summary or {}, indent=2)}

        Your Task:
        1) Decompose the problem into an ordered sequence of **unique** primitives (<= {MAX_PRIMITIVES}).
        2) Reuse available primitives where possible (only include id/name/status for reuses).
        3) Define new primitives fully (all fields required).
        4) Always include a final Evaluation primitive to terminate.
        5) Output strictly the JSON schema and finish with the token <END_OF_SEQUENCE>.
        """
    
    print("Calling LLM to generate primitive sequence...")

    complexity_estimate = len(tokenizer(system_prompt + user_prompt)['input_ids'])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate))

    primitives_sequence = generate_text(
        model, tokenizer, system_prompt, user_prompt,
        dynamic_max_tokens=dynamic_max_tokens
    )["primitive_sequence"]

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







