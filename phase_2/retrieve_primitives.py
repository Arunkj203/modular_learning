# retrieve_primitives.py

import json
from config import PRIMITIVE_LIBRARY_PATH


# --- Stage B: Primitive Retrieval ---

def retrieve_relevant_primitives(library_path, analysis):
    """
    Retrieve primitives relevant to the problem analysis.

    Args:
        library_path (str): Path to primitive library (JSON or DB)
        analysis (dict): Analysis output from Phase 1 containing:
                         - problem_type
                         - domain
                         - methods
                         - tags

    Returns:
        list: Relevant primitives (list of dicts)
    """
    try:
        with open(library_path, "r") as f:
            primitive_library = json.load(f)
    except FileNotFoundError:
        print(f"No primitive library found at {library_path}. Returning empty list.")
        return []

    relevant = []

    for primitive in primitive_library:
        # Check if primitive matches the domain or methods or tags
        if "domain" in analysis and primitive.get("domain") == analysis["domain"]:
            relevant.append(primitive)
        elif "methods" in analysis and any(m in primitive.get("methods", []) for m in analysis["methods"]):
            relevant.append(primitive)
        elif "tags" in analysis and any(t in primitive.get("tags", []) for t in analysis["tags"]):
            relevant.append(primitive)

    return relevant
