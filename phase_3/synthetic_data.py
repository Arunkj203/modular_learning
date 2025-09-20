# phase_3/synthetic_data.py
from models_config import call_openrouter
import json

# === Seed Example Generation ===
def generate_seed_examples(primitive_entry, n=5):
    """
    Generate a few seed examples for a primitive using LLM.
    Each example has {"input": ..., "output": ...}.
    """
    system_prompt = "You are a helpful data generator that produces high-quality training examples."
    user_prompt = f"""
    Primitive:
    ID: {primitive_entry['id']}
    Description: {primitive_entry.get('description', '')}
    Input schema: {primitive_entry.get('input', {})}
    Output schema: {primitive_entry.get('output', {})}

    Task:
    Generate {n} synthetic training examples as a JSON list.
    Each example must have "input" and "output" fields.
        """

    response = call_openrouter(system_prompt, user_prompt)
    return parse_json_list(response)


# === Bootstrapping from Seeds ===
def bootstrap_examples(seed_examples, target_size=30):
    """
    Expand seed examples into a larger dataset by asking LLM
    to generate paraphrased or varied versions.
    """
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
            """

            response = call_openrouter(system_prompt, user_prompt)
            new_ex = parse_json_obj(response)
            if new_ex:
                examples.append(new_ex)

    return examples


# === JSON Helpers ===
def parse_json_list(text: str):
    """
    Try to parse LLM output into a Python list of dicts.
    Cleans common formatting issues.
    """
    try:
        if text.startswith("```"):
            text = text.strip("`").split("json")[-1].strip()
        return json.loads(text)
    except Exception as e:
        print("⚠️ JSON list parsing failed, returning empty list. Error:", e)
        return []


def parse_json_obj(text: str):
    """
    Try to parse LLM output into a single JSON object (dict).
    Cleans common formatting issues.
    """
    try:
        if text.startswith("```"):
            text = text.strip("`").split("json")[-1].strip()
        return json.loads(text)
    except Exception as e:
        print("⚠️ JSON object parsing failed, skipping. Error:", e)
        return None


# === Main Function ===
def generate_synthetic_data_for_primitive(primitive_entry, target_size=30):
    """
    Generate a synthetic dataset for a primitive by:
    1. Creating seed examples
    2. Bootstrapping them to target size
    """
    seeds = generate_seed_examples(primitive_entry, n=5)
    dataset = bootstrap_examples(seeds, target_size=target_size)
    return dataset
