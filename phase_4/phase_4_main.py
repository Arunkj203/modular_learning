# phase_4_main.py
import torch
import json
from peft import PeftModel


from ..model_config import OUTPUT_DIR, DEVICE , call_openrouter ,generate_text

def run_phase4(base_model, tokenizer  ,primitive_sequence, problem_text,use_lora=False):

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
    feedback_entries = []  # Collect feedback for this problem

    # Cache for loaded LoRA models
    primitive_cache = {}

    
    for primitive_entry in primitive_sequence:
        primitive_id = primitive_entry["id"]
        primitive_name = primitive_entry.get("name", "")
        description = primitive_entry.get("description", "")

        if use_lora :

            try:
                # === Load or reuse LoRA adapter ===
                if primitive_id not in primitive_cache:
                    lora_path = f"{OUTPUT_DIR}/{primitive_id}"
                    primitive_model = PeftModel.from_pretrained(base_model, lora_path)
                    primitive_model.to(DEVICE)
                    primitive_model.eval()
                    primitive_cache[primitive_id] = primitive_model
                model = primitive_cache[primitive_id]
                
                # === Inference ===
                inputs = tokenizer(state_text, return_tensors="pt", truncation=True).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=128)
                primitive_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # === Validate & correct ===
                judge_result = llm_validate_and_correct(
                    primitive_id,
                    description,
                    state_text,
                    primitive_output
                )

                corrected_output = judge_result.get("corrected_output", primitive_output)
                valid = judge_result.get("valid", False)

                steps.append({
                    "primitive": primitive_name,
                    "description": description,
                    "input": state_text,
                    "output": primitive_output,
                    "valid": valid,
                    "validation_reason": judge_result.get("reason", ""),
                    "corrected_output": corrected_output
                })

                feedback_entries.append({
                    "primitive_id": primitive_id,
                    "input": state_text,
                    "output": primitive_output,
                    "valid": valid,
                    "validation_reason": judge_result.get("reason", ""),
                    "corrected_output": corrected_output
                })

                # === Update state for next primitive ===
                state_text = corrected_output

            except Exception as e:
                print(f"Error applying primitive {primitive_id}: {e}")
                return None, steps, feedback_entries


        else:
           # Build system and user prompts for primitive execution
            system_prompt = """You are a precise executor of primitive operations.
                    Always apply the given primitive to the problem state.
                    Always transform the state logically or numerically as instructed.
                    Never repeat or restate the question text.
                    Always return only the computed new state as JSON wrapped in <start> and <end>.
                    """

            user_prompt = f"""
                Problem State:
                {state_text}

                Primitive to apply:
                ID: {primitive_id}
                Name: {primitive_name}
                Description: {description}

                Task: Apply the primitive operation to the problem state.
                If numbers are involved, compute them. If expressions are involved, simplify them.
                Return JSON in this exact format:

                <start>
                {{
                "result": "updated state or computed value"
                }}
                <end>
                """

            
            # Calculate dynamic max_tokens based on complexity
            complexity_estimate = len(tokenizer(user_prompt)['input_ids'])
            dynamic_max_tokens = min(4096, max(512, 2 * complexity_estimate )) 

                # Call your generate_text wrapper
            op = generate_text(
                    model=base_model, 
                    tokenizer=tokenizer, 
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt,
                    dynamic_max_tokens=dynamic_max_tokens
                )
                
            # Record this step (include pre/post state for debugging)
            steps.append({
                "primitive_id": primitive_id,
                "name": primitive_name,
                "description": description,
                "input": state_text,
                "output": op
            })

            # Update the state for the next primitive
            state_text = op

    return state_text, steps , feedback_entries


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
