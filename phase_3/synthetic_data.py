import json , re
from ..model_config import generate_text

# === Robust JSON Parser ===
def parse_raw_op_with_markers(raw_text: str):
    """
    Extract JSON array of primitives from raw LLM output wrapped with <start> and <end>
    """
    # Extract text between <start> and <end>
    match = re.search(r"<start>(.*?)<end>", raw_text, flags=re.S)
    if not match:
        raise ValueError("Could not find <start> ... <end> in raw output")

    json_text = match.group(1).strip()

    # Remove trailing commas before } or ]
    json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)

    # Parse JSON
    return json.loads(json_text)


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

        Input Schema:
        {primitive_entry.get('input_schema', {})}

        Output Schema:
        {primitive_entry.get('output_schema', {})}

        Requirements:
        - Generate at least 5 diverse examples with different types of inputs (edge cases, negative, zero, etc.).
        - Ensure outputs are **correctly computed** from the inputs.
        - Format as a JSON list of objects:
        - Enclose the JSON array between <start> and <end>.

        [
        {{
        "input": {primitive_entry.get('input_schema', {})},
        "output": {primitive_entry.get('output_schema', {})}
        }},
        ...
        ]
        - Make it neat and consistent for training.
        """

    response = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=1500)
    print(response)

    return parse_raw_op_with_markers(response)


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
- Enclose the JSON array between <start> and <end>.
- Output only valid JSON.
- No extra text outside the JSON.
"""

            response = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=500)
            new_ex = parse_raw_op_with_markers(response)
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

        print(f"Generated seeds:\n {seeds} seed examples for format {f}.")
        format_dataset = bootstrap_examples(model, tokenizer, seeds, target_size=n_samples_per_format)

        print(f"Bootstrapped data:\n {format_dataset} examples for format {f}.")

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
