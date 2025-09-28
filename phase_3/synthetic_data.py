import json
from ..model_config import generate_text

# === Robust JSON Parser ===
def parse_json(text: str, expect_list: bool = False):
    """
    Extract and parse JSON from LLM output.
    Cleans up markdown fences, stray text, and ensures valid JSON.

    Args:
        text (str): raw LLM output
        expect_list (bool): True if expecting a JSON array

    Returns:
        parsed (dict or list): parsed JSON object or list
    """
    try:
        # Remove markdown fences (```json ... ```)
        if text.startswith("```"):
            text = text.strip("`").split("json")[-1].strip()

        # Extract JSON substring (between first { or [ and last } or ])
        start = min([i for i in [text.find("{"), text.find("[")] if i != -1])
        end = max([i for i in [text.rfind("}"), text.rfind("]")] if i != -1]) + 1
        json_str = text[start:end].strip()

        parsed = json.loads(json_str)

        # If expecting list but got object, wrap it
        if expect_list and isinstance(parsed, dict):
            parsed = [parsed]

        return parsed

    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {e}\nRaw text: {text}")


# === Seed Example Generation ===
def generate_seed_examples_for_format(model, tokenizer, primitive_entry, format_id, n=5):
    """
    Generate n seed training examples for a primitive in a specific format.
    """
    system_prompt = "You are a helpful data generator that produces high-quality training examples."

    user_prompt = f"""
Primitive:
ID: {primitive_entry['id']}
Name: {primitive_entry['name']}
Description: {primitive_entry.get('description', '')}
Domain: {primitive_entry.get('domain', '')}
Input constraints: {primitive_entry.get('constraints', '')}

Task:
Create {n} DISTINCT training examples in JSON.


Valid output format example (for 2 examples):
[
  {{"input": "some input text", "output": "expected output text"}},
  {{"input": "another input", "output": "another output"}}
]

Rules:
- Output MUST be ONLY a JSON array of {n} objects.
- Each object must have exactly two fields: "input" (string) and "output" (string).
- No explanations, no markdown, no extra text.
- Ensure all examples follow the primitive's description and constraints.

"""

    response = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=1500)
    
    return parse_json(response, expect_list=True)


# === Bootstrapping from Seeds ===
def bootstrap_examples(model, tokenizer, seed_examples, target_size=20):
    """
    Expand seed examples into a larger dataset by paraphrasing.
    """
    system_prompt = "You are a helpful data generator that creates paraphrased variations of training examples."
    examples = seed_examples.copy()

    while len(examples) < target_size:
        for ex in seed_examples:
            if len(examples) >= target_size:
                break

            user_prompt = f"""
Given this example:
{json.dumps(ex, ensure_ascii=False)}

Generate a paraphrased variant with the same meaning,
but different phrasing, numbers, or structure.
Return only valid JSON with "input" and "output".

Important:
- Output only valid JSON.
- No extra text outside the JSON.
"""

            response = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=500)
            new_ex = parse_json(response, expect_list=False)
            if new_ex:
                examples.append(new_ex)

    return examples


# === Main Function ===
def generate_synthetic_data_for_primitive(
    model, tokenizer, primitive_entry, n_formats=3, n_samples_per_format=20, save_path=None
):
    """
    Generate a synthetic dataset for a primitive with:
    - n_formats different input-output formats
    - n_samples_per_format per format
    Returns a list of dicts with "text" field, directly usable in LoRA training.
    """
    full_dataset = []

    for f in range(1, n_formats + 1):
        print(f"=== Generating format {f} for primitive {primitive_entry['id']} ===")
        seeds = generate_seed_examples_for_format(model, tokenizer, primitive_entry, f, n=5)

        print(f"Generated \n {seeds} seed examples for format {f}.")
        format_dataset = bootstrap_examples(model, tokenizer, seeds, target_size=n_samples_per_format)

        # Convert to LoRA training format
        for ex in format_dataset:
            text_example = f"Input: {ex['input']}\nOutput: {ex['output']}"
            full_dataset.append({"text": text_example, "format_id": f})

    # Optionally save
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(full_dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved dataset to {save_path}")

    return full_dataset
