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
# Retrieval
# ---------------------------------------------------------------------

def retrieve_primitives(
    analysis,
    top_k=10,
    expand_related=True,
    depth=1,
    min_similarity=0.5,
    using_l2=True,
):
    """
    Retrieve semantically relevant primitives from memory using FAISS and graph expansion.

    Args:
        analysis (dict): Analysis output of the problem (topics, tags, modules, etc.)
        top_k (int): Number of top similar primitives to retrieve.
        expand_related (bool): Whether to expand retrieval using primitive graph neighbors.
        depth (int): Depth for graph expansion.
        min_similarity (float): Minimum similarity threshold.
        using_l2 (bool): Whether FAISS index is built using L2 distance (else inner product).

    Returns:
        List[dict]: Retrieved primitive metadata sorted by adjusted similarity.
    """

    # --- Safety check ---
    if mem.faiss_index is None or mem.faiss_index.ntotal == 0:
        print("⚠️ No primitives in memory index yet.")
        return []

    # --- Step 1: Build a semantically rich query embedding ---
    modules = analysis.get("selected_modules", [])
    topics = analysis.get("topics", [])
    tags = analysis.get("tags", [])
    problem_type = analysis.get("problem_type", "")

    embeddings, weights = [], []

    # Give higher weight to problem type and module descriptions
    if problem_type:
        embeddings.append(mem.embed_model.encode(problem_type))
        weights.append(2.0)

    for t in topics:
        embeddings.append(mem.embed_model.encode(t))
        weights.append(1.0)

    for tag in tags:
        embeddings.append(mem.embed_model.encode(tag))
        weights.append(0.7)

    for m in modules:
        desc = f"{m['name']}: {m['description']}"
        embeddings.append(mem.embed_model.encode(desc))
        weights.append(1.5)

    if not embeddings:
        print("⚠️ Empty analysis; cannot retrieve primitives.")
        return []

    # Weighted average embedding (normalized)
    query_vec = np.average(embeddings, axis=0, weights=weights).astype("float32")
    query_vec /= np.linalg.norm(query_vec) + 1e-8

    # --- Step 2: Search in FAISS ---
    D, I = mem.faiss_index.search(np.array([query_vec]), top_k)

    retrieved = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        pid = mem.primitive_id_map.get(idx)
        if not pid or pid not in mem.primitive_metadata:
            continue

        # Convert FAISS distance to similarity score
        if using_l2:
            sim_score = max(0.0, 1 - dist / 2)  # approximate cosine conversion
        else:
            sim_score = (1 + dist) / 2  # inner product to 0–1 range

        if sim_score < min_similarity:
            continue

        prim = dict(mem.primitive_metadata[pid])  # copy to avoid mutating global store
        prim["similarity_score"] = float(sim_score)
        retrieved.append(prim)

    # --- Step 3: Optional graph expansion ---
    if expand_related and retrieved:
        expanded = set(p["id"] for p in retrieved)
        frontier = set(expanded)

        for _ in range(depth):
            next_frontier = set()
            for pid in frontier:
                if pid not in mem.primitive_graph:
                    continue
                for n in mem.primitive_graph.neighbors(pid):
                    if n not in expanded:
                        next_frontier.add(n)
            expanded.update(next_frontier)
            frontier = next_frontier

        for pid in expanded:
            if pid not in [p["id"] for p in retrieved] and pid in mem.primitive_metadata:
                neighbor_prim = mem.primitive_metadata[pid]
                neighbor_prim = dict(neighbor_prim)
                neighbor_prim["similarity_score"] = 0.3  # default for expanded nodes
                retrieved.append(neighbor_prim)

    # --- Step 4: Adjust ranking by connectivity (optional bias for generality) ---
    for p in retrieved:
        degree = len(list(mem.primitive_graph.neighbors(p["id"]))) if p["id"] in mem.primitive_graph else 0
        p["adjusted_score"] = p["similarity_score"] + 0.01 * np.log1p(degree)

    retrieved.sort(key=lambda x: -x.get("adjusted_score", 0.0))

    # --- Step 5: Clean structured output ---
    output = [
        {
            "id": p["id"],
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "goal": p.get("goal", ""),
            "type": p.get("primitive_type", ""),
            "score": round(p.get("similarity_score", 0.0), 3),
        }
        for p in retrieved
    ]

    return output


# ===============================================================
# STEP 2 — Evaluate sufficiency of retrieved primitives
# ===============================================================
def evaluate_primitive_sufficiency(model, tokenizer, problem_text, analysis, retrieved):
    """
    Ask the LLM whether the retrieved primitives are sufficient to solve
    the given problem. If not, it must describe missing capabilities.
    """

    available_str = json.dumps(
        [{"id": p["id"], "name": p["name"], "desc": p["description"]} for p in retrieved],
        indent=2
    )

    modules = analysis.get("selected_modules", [])
    modules_str = "; ".join([f"{m['name']}: {m['description']}" for m in modules])

    system_prompt = """
You are an analytical reasoning expert evaluating a set of available cognitive primitives.

Your job:
1. Determine whether the listed primitives are sufficient to solve the problem type.
2. If they are sufficient, list which ones can be reused.
3. If they are not sufficient, explain what conceptual capability or reasoning operation is missing.

Respond in valid JSON with this schema:
{
  "reuse": ["<primitive_ids>"],
  "missing_capabilities": ["<conceptual gaps, if any>"]
}
<END_OF_EVALUATION>
"""

    user_prompt = f"""
Problem Type: {analysis.get('problem_type', 'Unknown')}
Topics: {', '.join(analysis.get('topics', []))}
Modules: {modules_str}

AVAILABLE_PRIMITIVES:
{available_str}

Problem Summary:
{problem_text}
"""

    print("🔎 Evaluating sufficiency of retrieved primitives...")

    raw = generate_text(
        model,
        tokenizer,
        system_prompt,
        user_prompt,
        dynamic_max_tokens=1000
    )

    try:
        result = json.loads(raw)
    except Exception:
        print("⚠️ Could not parse LLM output, fallback to empty evaluation.")
        result = {"reuse": [], "missing_capabilities": []}

    return result


# ===============================================================
# STEP 3 — Generate missing primitives & final sequence
# ===============================================================
def generate_primitives_with_reflection(
    model,
    tokenizer,
    problem_text,
    analysis,
    retrieved,
    sufficiency_result,
    max_primitives=12
):
    """
    Create new primitives only if required and construct a full ordered reasoning sequence.
    """

    # -------------------------------
    # Build system + user prompts
    # -------------------------------
    system_prompt = """
You are a reasoning planner that constructs an abstract sequence of reusable cognitive primitives.

Rules:
1. Reuse existing primitives whenever possible.
2. Only create new primitives if the sufficiency analysis identified missing capabilities.
3. Each primitive represents a general mental operation.
4. No numbers, names, or domain entities.
5. Always end with an Evaluation primitive.
6. Output valid JSON using this format:

{
  "primitive_sequence": [
    {"step": 1, "id": "<Existing ID>", "name": "<Primitive Name>", "status": "Existing"},
    {"step": 2, "id": "P_new###", "name": "<New Primitive>", "status": "New"},
    ...
  ]
}
<END_OF_SEQUENCE>
"""

    reuse_str = json.dumps(sufficiency_result.get("reuse", []), indent=2)
    missing_str = json.dumps(sufficiency_result.get("missing_capabilities", []), indent=2)

    available_str = json.dumps(
        [{"id": p["id"], "name": p["name"], "desc": p["description"]} for p in retrieved],
        indent=2
    )

    modules = analysis.get("selected_modules", [])
    modules_str = "; ".join([f"{m['name']}: {m['description']}" for m in modules])

    user_prompt = f"""
Problem Type: {analysis.get('problem_type', 'Unknown')}
Topics: {', '.join(analysis.get('topics', []))}
Modules: {modules_str}

AVAILABLE_PRIMITIVES:
{available_str}

Reuse these primitives if possible:
{reuse_str}

Conceptual capabilities missing:
{missing_str}

Task:
Plan an ordered sequence (≤ {max_primitives}) of primitives to solve the problem.
Use the missing capabilities only to justify new primitives.
Do NOT mention concrete values or story details.
"""

    print("🧩 Generating final primitive plan...")

    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate))

    result = generate_text(
        model,
        tokenizer,
        system_prompt,
        user_prompt,
        dynamic_max_tokens=dynamic_max_tokens
    )

    try:
        primitives_sequence = result["primitive_sequence"]
    except Exception:
        raise ValueError("LLM did not return valid primitive_sequence JSON.")

    # -------------------------------
    # Clean & register new primitives
    # -------------------------------
    valid_ids = set(mem.primitive_metadata.keys())
    clean_seq, seen = [], set()

    for p in primitives_sequence:
        pid = p.get("id")
        name = p.get("name", "").strip()
        if not pid or not name or pid in seen:
            continue
        seen.add(pid)

        if pid in valid_ids:
            p["status"] = "Existing"
        else:
            # If model thinks it's existing but we don't have it
            if p.get("status") == "Existing":
                print(f"[WARN] Primitive {pid} not found; marking as New.")
            p["id"] = f"P_new_{uuid.uuid4().hex[:5]}"
            p["status"] = "New"
            add_primitive(p)

        clean_seq.append(p)

    # Guarantee an evaluation step at the end
    if not any("EVALUATE" in p["name"].upper() for p in clean_seq):
        eval_pid = (
            [pid for pid, prim in mem.primitive_metadata.items() if "EVALUATE" in prim["name"].upper()] or ["P001"]
        )[0]
        clean_seq.append({
            "step": len(clean_seq) + 1,
            "id": eval_pid,
            "name": "EVALUATE_RESULT",
            "status": "Existing"
        })

    return clean_seq




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








