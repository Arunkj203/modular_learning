# analyze_problems.py

import json 
from typing import Dict, Any

from models_config.model_config import call_openrouter
from utils import extract_json_from_text

def analyze_problem(problem_entry: Dict[str, Any]) -> Dict[str, Any]:

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

    system_prompt = "You are an expert problem analyst. Identify the problem type, domain, and reasoning strategies."
    user_prompt = f"""
    Problem: {question}

    Intermediate steps (if any): {steps}

    Return a JSON object with fields:
    - problem_type
    - domain
    - methods
    - tags
    """

    raw = call_openrouter(system_prompt, user_prompt, max_tokens=400)

    try:
        json_text = extract_json_from_text(raw)
        return json.loads(json_text)
    
    except Exception:
        
        print("⚠️ Could not parse JSON from analysis output:", raw)
        return {"problem_type": "unknown", "domain": "unknown", "methods": [], "tags": []}