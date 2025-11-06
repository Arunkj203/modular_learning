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

Rules:
1. Reuse existing primitives whenever possible.
2. Only create new primitives if the sufficiency analysis identified missing capabilities.
3. Each primitive represents a general mental operation.
4. No numbers, names, or domain entities.
5. Always end with an Evaluation primitive.
6. For new primitives, include a brief description explaining their cognitive role or reasoning function.
7. Output valid JSON using this format:

{
  "primitive_sequence": [
    {"step": 1, "id": "<Existing ID>", "name": "<Primitive Name>", "status": "Existing"},
    {"step": 2, "id": "P_new###", "name": "<New Primitive>", "description": "<What this new primitive does>", "status": "New"},
    ...
  ]
}
<END_OF_SEQUENCE>
"""


    reuse_str = json.dumps(sufficiency_result.get("reuse", []), indent=2)
    missing_str = json.dumps(sufficiency_result.get("missing_capabilities", []), indent=2)
    # print("Reuse:",reuse_str)
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

    # print(primitives_sequence)
    return clean_and_register_primitives(primitives_sequence)

def clean_and_register_primitives(primitives_sequence, similarity_threshold=0.9):
    """
    Deduplicate, merge, and register primitives from a generated sequence.

    - Reuses existing primitives if semantically similar.
    - Assigns new IDs only when no close match exists.
    - Guarantees a final EVALUATE primitive.

    Args:
        primitives_sequence (list[dict]): Raw LLM-generated primitive list.
        similarity_threshold (float): Cosine similarity threshold for merging.

    Returns:
        list[dict]: Cleaned and finalized primitive sequence.
    """

    if not primitives_sequence:
        print("No primitives provided to clean_and_register_primitives.")
        return []

    valid_ids = set(mem.primitive_metadata.keys())
    clean_seq, seen = [], set()

    for p in primitives_sequence:
        pid = p.get("id")
        name = p.get("name", "").strip()

        if not pid or not name or pid in seen:
            continue
        seen.add(pid)

        # ------------------------------------------------------------------
        # STEP 1 — Check if primitive already exists by ID
        # ------------------------------------------------------------------
        if pid in valid_ids:
            p["status"] = "Existing"
            clean_seq.append(p)
            continue

        # ------------------------------------------------------------------
        # STEP 2 — Semantic deduplication (find nearest existing primitive)
        # ------------------------------------------------------------------
        text_repr = f"{name}. {p.get('description', '')}"
        new_vec = mem.embed_model.encode(text_repr).astype("float32")
        new_vec /= np.linalg.norm(new_vec) + 1e-8

        # Retrieve top-k closest existing primitives
        if mem.faiss_index and mem.faiss_index.ntotal > 0:
            D, I = mem.faiss_index.search(np.array([new_vec]), k=5)
            best_match = None
            best_score = -1

            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                eid = mem.primitive_id_map.get(idx)
                if not eid or eid not in mem.primitive_metadata:
                    continue
                existing_prim = mem.primitive_metadata[eid]

                # Convert FAISS distance → cosine similarity
                sim = 1 - (dist / 2)
                if sim > best_score:
                    best_score = sim
                    best_match = existing_prim

            if best_match and best_score >= similarity_threshold:
                # Merge with existing primitive
                p["id"] = best_match["id"]
                p["name"] = best_match["name"]
                p["description"] = best_match.get("description", "")
                p["status"] = "Existing"
                p["merged_from"] = name
                print(f"[MERGE] '{name}' → existing primitive '{best_match['name']}' (sim={best_score:.2f})")
                clean_seq.append(p)
                continue

        # ------------------------------------------------------------------
        # STEP 3 — If no close match, register as a new primitive
        # ------------------------------------------------------------------
        p["id"] = f"P_new_{uuid.uuid4().hex[:5]}"
        p["status"] = "New"

        if "description" not in p or not p["description"]:
            p["description"] = f"Auto-generated primitive: {name}"

        clean_seq.append(p)
        # print(f"[NEW] Added new primitive: {p['id']} — {name}")

    # ----------------------------------------------------------------------
    # STEP 4 — Guarantee at least one Evaluation primitive at end
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # STEP 5 — Return cleaned sequence
    # ----------------------------------------------------------------------
    for i, p in enumerate(clean_seq, start=1):
        p["step"] = i  # Ensure sequential numbering

    return clean_seq


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






