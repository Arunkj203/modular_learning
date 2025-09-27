
import json,re
from typing import Dict, Any, List

# ---------------- Parsing Helpers ----------------

def extract_json_from_text(text: str) -> str:
    """
    Extract the first valid JSON object or array from LLM output.
    Handles extra commentary, backticks, or partial echoes.
    """

     # Try to find RESPONSE: first
    match = re.search(r'RESPONSE:\s*(.*)', text, flags=re.S)
    if match:
        response_text = match.group(1)
    else:
        response_text = text  # fallback: assume entire output is JSON

        # Remove Markdown code fences
    response_text = re.sub(r'```(?:json)?', '', response_text)

    # Try direct json.loads
    try:
        json.loads(response_text)
        return response_text
    except Exception:
        pass


    # Bracket matching
    start_idx = None
    start_char = None
    for i, ch in enumerate(response_text):
        if ch in ('{', '['):
            start_idx = i
            start_char = ch
            break
    if start_idx is None:
        raise ValueError("No JSON object/array found in LLM output.")

    stack = []
    for j in range(start_idx, len(response_text)):
        ch = response_text[j]
        if ch == start_char:
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
        if not stack:
            candidate = response_text[start_idx:j+1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue

    # Last resort: non-greedy regex
    matches = re.findall(r'(\[.*?\]|\{.*?\})', response_text, flags=re.S)
    for m in matches:
        try:
            json.loads(m)
            return m
        except Exception:
            continue

    raise ValueError("Failed to parse JSON from LLM output.")




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


