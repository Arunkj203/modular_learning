import json

def llm_validate_step(judge_model, primitive_id, desc, input_text, output_text):
    """
    Uses an LLM to validate if a primitive output is correct.
    """
    prompt = f"""
    Primitive ID: {primitive_id}
    Description: {desc}
    Input: {input_text}
    Output: {output_text}

    Question: Does the output correctly follow the description given the input?
    Respond with JSON: {{"valid": true/false, "reason": "short explanation"}}

    Important:
    - Output only valid JSON.
    - Do not include any extra text or code after the JSON.
    - Stop immediately after closing the final brace of the JSON object.

    """
    response = judge_model.generate(prompt, max_new_tokens=128)
    # assume response is parseable JSON
    try:
        import json
        return json.loads(response)
    except Exception:
        return {"valid": None, "reason": "Judge response unparsable"}
