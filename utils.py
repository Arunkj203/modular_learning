
import json,os,re
from typing import Dict, Any, List

# ---------------- Parsing Helpers ----------------
def extract_json_from_text(text: str) -> str:
    """
    Attempts to extract the first JSON object/array from text.
    Helpful when the LLM adds backticks or commentary.
    """
    # Try direct json loads first
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Find first { ... } or [ ... ] block using bracket matching
    start_idx = None
    start_char = None
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            start_idx = i
            start_char = ch
            break
    if start_idx is None:
        raise ValueError("No JSON object/array found in LLM output.")

    # find matching bracket
    stack = []
    for j in range(start_idx, len(text)):
        ch = text[j]
        if ch == start_char:
            stack.append(ch)
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            else:
                # mismatch
                pass
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        if not stack:
            candidate = text[start_idx:j+1]
            try:
                _ = json.loads(candidate)
                return candidate
            except Exception:
                continue
    # last resort: extract all braces and attempt
    matches = re.findall(r'(\[.*\]|\{.*\})', text, flags=re.S)
    for m in matches:
        try:
            _ = json.loads(m)
            return m
        except Exception:
            continue
    raise ValueError("Failed to parse JSON from LLM output.")


# --- JSON handling ---
def load_primitives(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_primitives(json_path: str, new_primitives: List[Dict[str, Any]]):
    """Append new primitives to JSON file without overwriting old ones."""
    all_primitives = load_primitives(json_path)
    all_primitives.extend(new_primitives)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_primitives, f, indent=4, ensure_ascii=False)


# ---------------- Validation ----------------
def validate_primitive_schema(p: Dict[str, Any]) -> List[str]:
    errors = []
    required = ["id", "name", "description", "input_schema", "output_schema", "example_call", "complexity", "reusable", "tags"]
    for r in required:
        if r not in p:
            errors.append(f"Missing required field: {r}")
    # id format
    if "id" in p and not re.match(r'^[a-z0-9\-]+$', p["id"]):
        errors.append("id must be lowercase letters, numbers, hyphens only")
    # basic schema types
    def check_schema(s, label):
        if not isinstance(s, dict):
            errors.append(f"{label} must be a JSON object describing fields")
    if "input_schema" in p:
        check_schema(p["input_schema"], "input_schema")
    if "output_schema" in p:
        check_schema(p["output_schema"], "output_schema")
    return errors


# --- Utility: Summarize old primitives ---
def summarize_primitives(primitives: List[Dict[str, Any]]) -> str:
    """Summarize old primitives to feed as context into the LLM prompt."""
    summaries = [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "tags": p.get("tags", []),
            "description": p.get("description", "")
        }
        for p in primitives
    ]
    return json.dumps(summaries, indent=2, ensure_ascii=False)



def convert_primitives_for_library(primitives: List[Dict]) -> List[Dict]:
    """
    Convert raw primitives into the Library Storage format with Metadata + LoRA (no path yet).
    """
    converted = []
    for p in primitives:
        entry = {
            "primitive_name": p.get("name", ""),
            "description": p.get("description", ""),
            "input_format": p.get("input_schema", {}),
            "output_format": p.get("output_schema", {}),
            "metadata": p.get("tags", [])
        }
        converted.append(entry)
    return converted


def print_primitives(primitives):
    """
    Print primitive details in a readable text format.
    """
    for p in convert_primitives_for_library(primitives):
        print("=== Primitive ===")
        print(f"Name       : {p['primitive_name']}")
        print(f"Description: {p['description']}")
        print(f"Input      : {json.dumps(p['input_format'], indent=2, ensure_ascii=False)}")
        print(f"Output     : {json.dumps(p['output_format'], indent=2, ensure_ascii=False)}")
        print(f"Metadata   : {', '.join(p['metadata']) if p['metadata'] else 'None'}")
        print("-----------------------END----------------------------------------------")




def export_primitives_summary(json_path: str, output_txt: str):
    """
    Load all primitives from the JSON library, summarize them,
    and save to a plain text file.
    """
    primitives = load_primitives(json_path)
    summary_text = summarize_primitives(primitives)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"✅ Saved summarized primitives to {output_txt}")
