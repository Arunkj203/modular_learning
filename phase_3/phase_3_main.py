# phase_4_main.py
import torch
import json
from peft import PeftModel

from ..config import *
from ..model_config import OUTPUT_DIR, DEVICE , call_openrouter ,generate_text


 # Build system and user prompts for primitive execution
system_prompt = """
You are a structured reasoning executor that applies human-like problem-solving primitives.

Your task is to transform the current problem state by applying the given primitive. 
Do not perform full problem solving — only apply the primitive’s transformation logic.

CRITICAL INSTRUCTIONS:
1. You must output ONLY valid JSON strictly between <<START>> and <<END>>.
2. JSON Format (no other text allowed):
{
    "result": "<new problem state after applying primitive>",
    "notes": "<short reasoning for transformation>"
}
3. Preserve all key details from the current state.
4. Apply the primitive exactly as described — don’t infer new rules or solve unrelated steps.
5. If the primitive involves symbolic or numeric manipulation, apply that change correctly and clearly.
6. Never produce explanations outside JSON or markers.

You are reasoning like a human who uses structured steps (primitives) to progressively modify the problem until solved.
"""


def run_phase3(base_model, tokenizer  ,primitive_sequence, problem_text):

    """
    Phase 4: Problem solving using a sequence of primitives (dicts with id, description).
    
    Args:
        primitive_sequence (list): List of dicts, each containing 'id' and 'description'.
        problem_text (str): Problem text to solve.
    
    Returns:
        final_solution (str), steps (list of dicts): Each dict has 'primitive_id', 'description', 'output'.
    """

    state_text = problem_text
    steps = []

    for idx,primitive_entry in enumerate(primitive_sequence):

        
        # primitive_entry = primitive_metadata.get(pid, {})
        if not primitive_entry:
            raise ValueError(f"Primitive ID {primitive_entry['id']} not found in metadata.")
        
        pid = primitive_entry.get("id", "")
        primitive_name = primitive_entry.get("name", "")
        description = primitive_entry.get("description", "")

       
        user_prompt = f"""

                You are given the current state of a problem and a primitive to apply.

                Problem State:
                {state_text}

                Apply primitive: "{primitive_entry['name']}" – {primitive_entry.get('description', '')}

                Now, generate the next problem state strictly following the system instructions.

                <<START>>
                {{
                "result": "...",
                "notes": "..."
                }}
                <<END>>

            **GENERATE THE NEXT STATE JSON NOW.**
            """

        
        # Calculate dynamic max_tokens based on complexity
        complexity_estimate = len(tokenizer(system_prompt + user_prompt)['input_ids'])
        dynamic_max_tokens = min(512, max(400, 2 * complexity_estimate )) 

            # Call your generate_text wrapper
        op = generate_text(
                model=base_model, 
                tokenizer=tokenizer, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt,
                dynamic_max_tokens=dynamic_max_tokens
            )
        
        
        # Record this step (include pre/post state for debugging)
        steps.append((op, primitive_name, description))

        # Update the state for the next primitive
        state_text = op["result"]
        print(f"Primitve {idx+1} output:\n{op}")
        
    return state_text, steps 


def llm_validate_and_correct(primitive_name, description, input_text, output_text):
    """
    Validate a primitive output and optionally correct it using an external judge model.

    Returns:
        dict: {
            "valid": True/False/None,
            "reason": "short explanation",
            "corrected_output": <corrected output if invalid, or original output if valid>
        }
    """
    system_prompt = "You are a strict JSON-only judge for validating primitive outputs."
    user_prompt = f"""
Primitive Name: {primitive_name}
Description: {description}
Input: {input_text}
Output: {output_text}

Question: Is the output correct? If not, provide the corrected output.

Respond ONLY with valid JSON in this format:
{{
    "valid": true/false,
    "reason": "short explanation",
    "corrected_output": "..."
}}
"""

    try:
        # Call OpenRouter judge
        response_text = call_openrouter(system_prompt, user_prompt, max_tokens=200)

        # Parse JSON response
        result = json.loads(response_text)

        # Ensure corrected_output is present
        if not result.get("corrected_output"):
            result["corrected_output"] = output_text

        return result

    except Exception as e:
        # Fallback if parsing fails
        return {
            "valid": None,
            "reason": f"LLM response unparsable: {str(e)}",
            "corrected_output": output_text
        }
















#  if use_lora :

#             try:
#                 # === Load or reuse LoRA adapter ===
#                 if primitive_id not in primitive_cache:
#                     lora_path = f"{OUTPUT_DIR}/{primitive_id}"
#                     primitive_model = PeftModel.from_pretrained(base_model, lora_path)
#                     primitive_model.to(DEVICE)
#                     primitive_model.eval()
#                     primitive_cache[primitive_id] = primitive_model
#                 model = primitive_cache[primitive_id]
                
#                 # === Inference ===
#                 inputs = tokenizer(state_text, return_tensors="pt", truncation=True).to(DEVICE)
#                 with torch.no_grad():
#                     outputs = model.generate(**inputs, max_new_tokens=128)
#                 primitive_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

#                 # === Validate & correct ===
#                 judge_result = llm_validate_and_correct(
#                     primitive_id,
#                     description,
#                     state_text,
#                     primitive_output
#                 )

#                 corrected_output = judge_result.get("corrected_output", primitive_output)
#                 valid = judge_result.get("valid", False)

#                 steps.append({
#                     "primitive": primitive_name,
#                     "description": description,
#                     "input": state_text,
#                     "output": primitive_output,
#                     "valid": valid,
#                     "validation_reason": judge_result.get("reason", ""),
#                     "corrected_output": corrected_output
#                 })

#                 feedback_entries.append({
#                     "primitive_id": primitive_id,
#                     "input": state_text,
#                     "output": primitive_output,
#                     "valid": valid,
#                     "validation_reason": judge_result.get("reason", ""),
#                     "corrected_output": corrected_output
#                 })

#                 # === Update state for next primitive ===
#                 state_text = corrected_output

#             except Exception as e:
#                 print(f"Error applying primitive {primitive_id}: {e}")
#                 return None, steps, feedback_entries
