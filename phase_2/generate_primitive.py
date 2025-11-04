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
        print("No primitives in memory index yet.")
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
        print("Empty analysis; cannot retrieve primitives.")
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

    result = generate_text(
        model,
        tokenizer,
        system_prompt,
        user_prompt,
        dynamic_max_tokens=1000
    )

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

Each primitive represents a general mental operation used to reason, transform, or evaluate information.

Rules:
1. Reuse existing primitives whenever possible (match by conceptual function).
2. Only create new primitives if the sufficiency analysis identified missing capabilities.
3. Avoid specific numbers, names, or story content.
4. Output valid JSON strictly in this format:

{
  "primitive_sequence": [
    {
      "step": 1,
      "id": "<Existing ID>",
      "name": "<Existing Primitive Name>",
      "status": "Existing"
    },
    {
      "step": 2,
      "id": "P_new###",
      "name": "<New Primitive Name>",
      "status": "New",
      "description": "<Conceptual role>",
      "input_types": ["<type1>", "<type2>"],
      "output_types": ["<type1>"],
      "category": "<Category>"
    }
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
Plan an ordered sequence (≤ {max_primitives}) of cognitive primitives to solve the problem.
Only create new primitives if they address a missing capability.
For existing primitives, include only id, name, and status.
For new primitives, include full metadata: description, input/output types, and category.
Do NOT mention concrete values or story details.
"""

    print("Generating final primitive plan...")

    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(1500, 2 * complexity_estimate))

    primitives_sequence = generate_text(
        model,
        tokenizer,
        system_prompt,
        user_prompt,
        dynamic_max_tokens=dynamic_max_tokens
    )["primitive_sequence"]

    return clean_and_register_primitives(primitives_sequence)


def add_primitive(primitive, semantic_threshold=0.8, top_k=5):
    """
    Add a primitive to memory, graph, and FAISS index.
    Builds semantic edges to similar primitives automatically.
    """

    pid = primitive["id"]
    mem.primitive_metadata[pid] = primitive

    # Build embedding from descriptive fields
    text = " ".join([
        primitive.get("name", ""),
        primitive.get("description", ""),
        primitive.get("domain", ""),
        primitive.get("problem_type", ""),
        " ".join(primitive.get("methods", [])),
        " ".join(primitive.get("tags", []))
    ]).strip()

    vec = mem.embed_model.encode(text).astype("float32")
    vec /= np.linalg.norm(vec) + 1e-8  # normalize for cosine similarity

    # Add node to graph
    mem.primitive_graph.add_node(pid, **primitive)

    # --- Optional: Semantic linking to similar primitives ---
    if mem.faiss_index and mem.faiss_index.ntotal > 0:
        D, I = mem.faiss_index.search(np.array([vec]), top_k)
        related = []

        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            sim = 1 - (dist / 2)  # convert FAISS L2 to cosine approx
            if sim < semantic_threshold:
                continue

            existing_pid = mem.primitive_id_map.get(idx)
            if existing_pid and existing_pid != pid:
                mem.primitive_graph.add_edge(
                    pid, existing_pid,
                    weight=sim,
                    relation="semantic"
                )
                mem.primitive_graph.add_edge(
                    existing_pid, pid,
                    weight=sim,
                    relation="semantic"
                )
                related.append(existing_pid)

        if related:
            primitive["related_primitives"] = related
            print(f"[LINK] {pid} semantically linked to {len(related)} primitives.")

    # Add to FAISS index
    idx = mem.faiss_index.ntotal
    mem.faiss_index.add(np.array([vec]))
    mem.primitive_id_map[idx] = pid

    return pid


def update_primitive_graph_from_sequence(sequence):
    """
    Create procedural edges between consecutive primitives in a reasoning sequence.
    """

    if not sequence or len(sequence) < 2:
        return

    for i in range(len(sequence) - 1):
        src = sequence[i]["id"]
        tgt = sequence[i + 1]["id"]

        if mem.primitive_graph.has_edge(src, tgt):
            mem.primitive_graph[src][tgt]["weight"] += 1
        else:
            mem.primitive_graph.add_edge(
                src, tgt,
                weight=1,
                relation="procedural"
            )

    print(f"[GRAPH] Added {len(sequence) - 1} procedural edges from current sequence.")

import uuid
import numpy as np

def clean_and_register_primitives(primitives_sequence, similarity_threshold=0.9):
    """
    Deduplicate, merge, validate, and register primitives from a generated sequence.

    Rules enforced:
      - Existing primitives in the LLM output must include ONLY: id, name, status.
      - New primitives must include: id (or placeholder), name, status, description,
          input_types, output_types, category.
      - Semantic similarity uses name + description.
      - Merge into closest existing primitive when sim >= similarity_threshold.
      - Register truly new primitives via register_primitive(p).

    Raises:
      ValueError: if the sequence contains invalid entries (missing or forbidden fields).
    Returns:
      list[dict]: cleaned, sequential primitive list.
    """

    if not primitives_sequence:
        print("No primitives provided to clean_and_register_primitives.")
        return []

    # memory helpers (assumed available in your environment)
    valid_ids = set(mem.primitive_metadata.keys())
    clean_seq = []
    seen = set()
    errors = []

    # helper sets
    allowed_existing_keys = {"id", "name", "status", "merged_from", "step"}
    required_new_keys = {"description", "input_types", "output_types", "category"}

    for raw_p in primitives_sequence:
        # shallow copy to avoid mutating caller's data
        p = dict(raw_p)

        pid = p.get("id")
        name = (p.get("name") or "").strip()
        status = (p.get("status") or "").strip()

        # basic sanity checks
        if not pid:
            errors.append(f"Primitive with name '{name}' missing 'id'.")
            continue
        if not name:
            errors.append(f"Primitive with id '{pid}' missing 'name'.")
            continue
        if pid in seen:
            errors.append(f"Duplicate primitive id in input sequence: '{pid}'.")
            continue
        seen.add(pid)

        # ---------------------------
        # Case A: It's an existing primitive (id present in memory)
        # ---------------------------
        if pid in valid_ids:
            # check that the generated entry doesn't include forbidden fields
            extra_keys = set(p.keys()) - allowed_existing_keys
            if extra_keys:
                errors.append(
                    f"Existing primitive '{pid}' ('{name}') must include only {allowed_existing_keys}. "
                    f"Found extra fields: {sorted(extra_keys)}"
                )
                # do not attempt to merge/normalize further for this entry
                continue

            # Use canonical metadata from memory
            meta = mem.primitive_metadata[pid]
            merged_entry = {
                "id": pid,
                "name": meta.get("name", name),
                "status": "Existing",
                # keep description empty in sequence to respect the user's rule (existing entries are lightweight)
            }
            clean_seq.append(merged_entry)
            continue

        # ---------------------------
        # Case B: It's a candidate new primitive (not in memory)
        # ---------------------------
        # Validate required fields for new primitives
        missing_new = [k for k in required_new_keys if not p.get(k)]
        if missing_new:
            errors.append(f"New primitive '{name}' (id={pid}) missing required fields: {missing_new}")
            continue

        # Normalize category string
        p["category"] = str(p["category"]).strip().capitalize()

        # Build text representation for semantic matching (name + description)
        text_repr = f"{name}. {p.get('description', '')}"
        try:
            new_vec = mem.embed_model.encode(text_repr).astype("float32")
            new_vec /= np.linalg.norm(new_vec) + 1e-8
        except Exception as e:
            errors.append(f"Embedding failure for '{name}': {e}")
            continue

        # semantic search against memory
        best_match = None
        best_score = -1.0
        if getattr(mem, "faiss_index", None) and mem.faiss_index.ntotal > 0:
            D, I = mem.faiss_index.search(np.array([new_vec]), k=5)
            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                eid = mem.primitive_id_map.get(idx)
                if not eid or eid not in mem.primitive_metadata:
                    continue
                existing_prim = mem.primitive_metadata[eid]
                sim = 1 - (dist / 2)  # convert FAISS distance to approximate cosine similarity
                if sim > best_score:
                    best_score = sim
                    best_match = existing_prim

        # If similar enough, merge as existing (but produced sequence must remain lightweight)
        if best_match and best_score >= similarity_threshold:
            merged = {
                "id": best_match["id"],
                "name": best_match.get("name", name),
                "status": "Existing",
                "merged_from": name
            }
            print(f"[MERGE] '{name}' → existing primitive '{best_match['name']}' (sim={best_score:.2f})")
            clean_seq.append(merged)
            continue

        # Otherwise, register as truly new primitive
        # If model provided an id that collides with memory or isn't following P_new pattern, reassign a unique id
        new_pid = pid
        if new_pid in valid_ids or not str(new_pid).startswith("P_new"):
            new_pid = f"P_new_{uuid.uuid4().hex[:8]}"
        p["id"] = new_pid
        p["status"] = "New"

        # Ensure types are lists
        if not isinstance(p.get("input_types"), list):
            p["input_types"] = [p["input_types"]]
        if not isinstance(p.get("output_types"), list):
            p["output_types"] = [p["output_types"]]

        
        clean_seq.append({
            "id": p["id"],
            "name": p["name"],
            "status": "New",
            # keep full metadata for new primitives in the returned sequence
            "description": p["description"],
            "input_types": p["input_types"],
            "output_types": p["output_types"],
            "category": p["category"]
        })

    # If any validation/registration errors occurred, raise a consolidated error
    if errors:
        raise ValueError("Primitive sequence validation/registration failed:\n" + "\n".join(errors))

    # Guarantee at least one Evaluation primitive at the end.
    def _is_evaluation(prim):
        # prim can be existing (lightweight) or new (has category)
        if prim.get("status") == "New":
            return prim.get("category", "").lower() == "evaluation"
        pid = prim.get("id")
        if pid in mem.primitive_metadata:
            return mem.primitive_metadata[pid].get("category", "").lower() == "evaluation" \
                   or "EVALUATE" in mem.primitive_metadata[pid].get("name", "").upper()
        return False

    if not any(_is_evaluation(p) for p in clean_seq):
        # find an existing evaluation primitive in memory
        eval_candidates = [pid for pid, meta in mem.primitive_metadata.items()
                           if meta.get("category", "").lower() == "evaluation" or "EVALUATE" in meta.get("name", "").upper()]
        eval_pid = eval_candidates[0] if eval_candidates else None
        if eval_pid:
            clean_seq.append({
                "id": eval_pid,
                "name": mem.primitive_metadata[eval_pid].get("name", "EVALUATE_RESULT"),
                "status": "Existing"
            })
        else:
            # if no evaluation primitive exists in memory, create a minimal new evaluation primitive and register it
            eval_pid = f"P_new_{uuid.uuid4().hex[:8]}"
            clean_seq.append({
                "id": eval_pid,
                "name": "EVALUATE_RESULT",
                "status": "New",
                "description": "Assess overall result for correctness and coherence.",
                "input_types": ["reasoning_trace"],
                "output_types": ["evaluation_score", "feedback"],
                "category": "Evaluation"
            })
            
            print(f"[NEW] Auto-registered evaluation primitive: {eval_pid}")

    # Re-index steps
    for i, prim in enumerate(clean_seq, start=1):
        prim["step"] = i

    return clean_seq





