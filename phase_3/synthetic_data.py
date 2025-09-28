from ..model_config import generate_text
import json

# === Seed Example Generation ===
def generate_seed_examples_for_format(model ,tokenizer, primitive_entry, format_id, n=5):
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

Format:
[
  {{
    "input": "string",
    "output": "string"
  }},
  ...
]

Rules:
- Each input must respect the constraints and demonstrate the transformation rule.
- Each output must show the correct transformed result.
- Return ONLY a valid JSON array of {n} objects.
- No extra text outside the JSON.
"""
    response = generate_text(model ,tokenizer, system_prompt, user_prompt)
    print(response)
    return parse_json_list(response)


# === Bootstrapping from Seeds ===
def bootstrap_examples(model ,tokenizer, seed_examples, target_size=20):
    system_prompt = "You are a helpful data generator that creates paraphrased variations of training examples."
    examples = seed_examples.copy()

    while len(examples) < target_size:
        for ex in seed_examples:
            if len(examples) >= target_size:
                break

            user_prompt = f"""
Given this example:
{ex}

Generate a paraphrased variant with the same meaning,
but different phrasing, numbers, or structure.
Return only valid JSON with "input" and "output".

Important:
- Output only valid JSON.
- Do not include any extra text or code after the JSON.
- Stop immediately after closing the final brace of the JSON object.

            """

            response = generate_text(model ,tokenizer, system_prompt, user_prompt)
            new_ex = parse_json_obj(response)
            if new_ex:
                examples.append(new_ex)

    return examples


# want to change these parsers to be more robust
# === JSON Helpers ===
def parse_json_list(text: str):
    try:
        if text.startswith("```"):
            text = text.strip("`").split("json")[-1].strip()
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"JSON list parsing failed:{e}")

def parse_json_obj(text: str):
    try:
        if text.startswith("```"):
            text = text.strip("`").split("json")[-1].strip()
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"JSON list parsing failed:{e}")


# === Main Function ===
def generate_synthetic_data_for_primitive(
   model ,tokenizer, primitive_entry, n_formats=3, n_samples_per_format=20, save_path=None
):
    """
    Generate a synthetic dataset for a primitive with:
    - n_formats different input-output formats
    - n_samples_per_format per format
    Returns a list of dicts with a "text" field, directly usable in LoRA training.
    """
    full_dataset = []

    for f in range(1, n_formats + 1):
        print(f"=== Generating format {f} for primitive {primitive_entry['id']} ===")
        seeds = generate_seed_examples_for_format(model ,tokenizer, primitive_entry, f, n=5)
        format_dataset = bootstrap_examples(model ,tokenizer, seeds, target_size=n_samples_per_format)

        # Convert to LoRA training format
        for ex in format_dataset:
            text_example = f"Input: {ex['input']}\nOutput: {ex['output']}"
            full_dataset.append({"text": text_example, "format_id": f})

    # Optionally save
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(full_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved dataset to {save_path}")

    return full_dataset
