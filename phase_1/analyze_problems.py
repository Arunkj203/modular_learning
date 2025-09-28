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

    system_prompt = "You are an expert problem analyst. Identify the problem type, domain, and reasoning strategies.You only produce JSON output.Never include code, explanations, or any text outside the JSON."
    user_prompt = f"""
    Problem: {question}

    Intermediate steps (if any): {steps}

    Return a JSON object with fields:
    - problem_type
    - domain
    - methods
    - tags

    Important:
    - Output only valid JSON.
    - Do not include any extra text or code after the JSON.
    - Stop immediately after closing the final brace of the JSON object.

    """

    raw = generate_text(model ,tokenizer, system_prompt, user_prompt, max_tokens=400)

    try:

        json_text = extract_analysis_dict(raw)
        return json.loads(json_text)
    
    except Exception as e:
        
        raise RuntimeError(f"Could not parse JSON from analysis output:{e}")
    


def extract_analysis_dict(raw_output: str) -> dict:
    """
    Extract the JSON dict after RESPONSE: from raw LLM output.
    """
    # Grab everything after RESPONSE:
    match = re.search(r'RESPONSE:\s*(\{.*?\})', raw_output, flags=re.S)
    if not match:
        # fallback: assume entire output is JSON
        response_text = raw_output
    else:
        response_text = match.group(1)

    # Remove potential backticks/code fences
    response_text = re.sub(r'```(?:json)?', '', response_text).strip()

    """
    Remove trailing commas before closing brackets/braces.
    """
    # Remove trailing commas before } or ]
    response_text = re.sub(r',(\s*[\}\]])', r'\1', response_text)

    # Parse JSON into Python dict
    return response_text
