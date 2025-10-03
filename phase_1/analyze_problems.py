# analyze_problems.py

from typing import Dict, Any

from ..model_config import generate_text
from ..config import parse_raw_op_with_markers , Retries


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
    "and break the problem into clear, sequential subtasks. "
    "Each subtask should describe **what action to perform** conceptually, "
    "not the solution itself. "
    "You must output a JSON object ONLY, and enclose it EXACTLY between <start> and <end> markers."
    )

    user_prompt = f"""
        Problem: {question}

        Intermediate steps (if any): {steps}


        Example:
        <start>
        {{
        "problem_type":"algebra",
        "domain":"math",
        "methods":["isolation","simplification"],
        "tags":["linear equation"],
        "subtasks":[
            {{"step":1,"instruction":"Identify the variable to isolate"}},
            {{"step":2,"instruction":"Move constants to the other side"}}
        ]
        }}
        <end>


        Instructions:
        - Output a valid JSON object with the following keys: problem_type, domain, methods, tags, subtasks.
        - Each subtask in 'subtasks' must have:
            - step: sequential number
            - instruction: a conceptual action describing what to do to progress toward solving the problem, **without computing the answer**.
        - Enclose the entire JSON exactly between <start> and <end> markers.
        - Do NOT include any extra text before, after, or inside the markers.

        """
    
        
        
    last_error = None
    for attempt in range(1, Retries + 1):
        raw = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=300)
        try:
            return parse_raw_op_with_markers(raw)
        except Exception as e:
            last_error = e
            print(f"[WARN] Attempt {attempt} failed: {e}")
            # optional: short delay before retry
    # If all attempts failed, raise
    raise RuntimeError(
        f"Could not parse JSON after {Retries} attempts. "
        f"Last error: {last_error}\nLast LLM output:\n{raw}"
    )
