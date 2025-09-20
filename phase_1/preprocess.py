import re

def preprocess_problem(problem_entry: dict, dataset_name: str = "SVAMP"):
    """
    Preprocess a math word problem entry into a unified format.
    
    Args:
        problem_entry (dict): One row from dataset.
        dataset_name (str): Which dataset (SVAMP, ASDiv, GSM8K, ...).
    
    Returns:
        dict: Unified format with keys:
              id, question, answer, intermediate_steps
    """
    dataset_name = dataset_name.lower()
    pid, question, answer, steps = None, None, None, None

    if dataset_name == "svamp":
        pid = problem_entry.get("ID")
        body = problem_entry.get("Body", "")
        q = problem_entry.get("Question", "")
        question = f"{body.strip()} {q.strip()}".strip()
        answer = problem_entry.get("Answer")
        # SVAMP has no explicit intermediate steps
        steps = None

    elif dataset_name == "asdiv":
        pid = problem_entry.get("ID")
        question = problem_entry.get("Problem", "").strip()
        answer = problem_entry.get("Answer")
        # equation field as steps if available
        steps = problem_entry.get("Equation")

    elif dataset_name == "gsm8k":
        pid = problem_entry.get("id")  # sometimes not available
        question = problem_entry.get("question", "").strip()
        answer = problem_entry.get("answer")
        steps = problem_entry.get("rationale") or problem_entry.get("steps")

    else:
        # fallback: try generic fields
        question = problem_entry.get("Problem") or problem_entry.get("question")
        answer = problem_entry.get("Answer") or problem_entry.get("answer")
        steps = problem_entry.get("Equation") or problem_entry.get("rationale")

    # Clean whitespace
    if question:
        question = re.sub(r"\s+", " ", question)
    if answer:
        answer = str(answer).strip()
    if steps:
        steps = re.sub(r"\s+", " ", str(steps))

    return {
        "id": pid,
        "question": question,
        "answer": answer,
        "intermediate_steps": steps
    }
