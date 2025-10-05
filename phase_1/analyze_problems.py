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
    "Break down the problem into conceptual subtasks that describe how to understand and represent the situation logically — not how to compute the answer. "
    "Each subtask should capture a reasoning action, like identifying quantities, relationships, or goals. "
    "Do not include algebraic manipulation steps or numeric calculations."
    )

    user_prompt = f"""
        Problem: {question}

        Intermediate steps (if any): {steps}


        Format your response like this (do not copy this text, only follow structure):
        <start>
            {{"problem_type": "...", "domain": "...", "methods": [...], "tags": [...], "subtasks":[...]}}
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

    # Calculate dynamic max_tokens based on complexity
    complexity_estimate = len(tokenizer(user_prompt)['input_ids'])
    dynamic_max_tokens = min(4096, max(300, 2 * complexity_estimate )) 


    for attempt in range(Retries):

        max_tokens = min(4096 , dynamic_max_tokens * (2 ** attempt) )        

        raw = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=300)
        try:
            return parse_raw_op_with_markers(raw)
        except Exception as e:
            last_error = e
            print(f"[WARN] Attempt {attempt+1} failed: {e}")
            # optional: short delay before retry
    # If all attempts failed, raise
    raise RuntimeError(
        f"Could not parse JSON after {Retries} attempts. "
        f"Last error: {last_error}\nLast LLM output:\n{raw}"
    )
