# analyze_problems.py

import json 
from typing import Dict, Any

import re


from ..model_config import generate_text
from ..config import parse_raw_op_with_markers


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
        "Identify the problem type, domain, reasoning strategies, "
        "and break the problem into clear, sequential subtasks."
        "You must output a JSON object ONLY, and enclose it EXACTLY between <start> and <end> markers."
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
        "tags": ["...","..."],
        "subtasks": [
            {{"step": 1, "instruction": "..."}},
            {{"step": 2, "instruction": "..."}}
        ]
    }}
    <end>

    Instructions:
    - Output a valid JSON object with the following keys: problem_type, domain, methods, tags, subtasks.
    - Subtasks should be a list of step/instruction objects, numbered sequentially.
    - Enclose the entire JSON **exactly** between <start> and <end> markers

    """

    
    
    
    raw = generate_text(model ,tokenizer, system_prompt, user_prompt, max_tokens=400)
    # print("Raw analysis output:", raw)
    try:
        return parse_raw_op_with_markers(raw)
    
    except Exception as e:
        raise RuntimeError(f"Could not parse JSON from analysis output:{e}\n LLM output:\n{raw}")
    

