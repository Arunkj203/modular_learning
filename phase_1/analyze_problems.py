# analyze_problems.py

import json 
from typing import Dict, Any

import re


from ..model_config import generate_text


def analyze_problem(model, tokenizer ,problem_entry: Dict[str, Any]) -> Dict[str, Any]:

    """
    Analyze a math problem using LLM via OpenRouter.
    
    Args:
        problem_entry: dict with keys from preprocessing:
                       {id, question, answer, intermediate_steps}
    Returns:
        dict with analysis fields
    """

    question = problem_entry.get("question", "")
    steps = problem_entry.get("intermediate_steps", "")

    
    system_prompt = (
    "You are an expert problem analyst. "
    "Identify the problem type, domain, and reasoning strategies."
    "You must output a JSON object ONLY, nothing else."
    )

    user_prompt = f"""
    Problem: {question}

    Intermediate steps (if any): {steps}

    Return exactly this JSON format enclosed in <start> and <end>:
    <start>

    {{
    "problem_type": "...",
    "domain": "...",
    "methods": ["...","..."],
    "tags": ["...","..."]
    }}

    <end>

    Rules:
    - Output ONLY valid JSON.
    - No extra keys.
    - No text before or after the JSON.
    """

    
    
    
    raw = generate_text(model ,tokenizer, system_prompt, user_prompt, max_tokens=400)
    # print("Raw analysis output:", raw)
    try:
        json_text = extract_analysis_dict(raw)
        return json.loads(json_text)
    
    except Exception as e:
        
        raise RuntimeError(f"Could not parse JSON from analysis output:{e}\n LLM output:\n{raw}")
    


def extract_analysis_dict(raw_output: str) -> dict:
     # Search for JSON object or array
    match = re.search(r'(\{.*?\}|\[.*?\])', raw_output, flags=re.S)
    if not match:
        raise ValueError("No JSON object/array found after RESPONSE:")

    json_text = match.group(1).strip()
    # Remove trailing commas
    json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)
    return json_text