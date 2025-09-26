# phase_4_main.py
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


from modular_learning.model_config import OUTPUT_DIR, DEVICE



def run_phase4(model, tokenizer , primitive_sequence, problem_text):
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

    # Load base model once (reuse for all primitives)
    base_model = model


    judge_model = AutoModelForCausalLM.from_pretrained("gpt-4-jurassic", device_map="auto")

    
    for primitive_entry in primitive_sequence:
        primitive_id = primitive_entry["id"]
        primitive_name = primitive_entry.get("name", "")
        description = primitive_entry.get("description", "")
        try:
            # Load LoRA adapter for this primitive
            lora_path = f"{OUTPUT_DIR}/{primitive_id}"
            model = PeftModel.from_pretrained(base_model, lora_path)
            model.eval()
            model.to(DEVICE)
            
            # Tokenize current state
            inputs = tokenizer(state_text, return_tensors="pt").to(DEVICE)
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            primitive_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # --- LLM validation & correction in a single step ---
            judge_result = llm_validate_and_correct(
                        judge_model,
                        primitive_id,
                        description,
                        state_text,
                        primitive_output
                    )


            steps.append({
                "primitive": primitive_name,
                "description": description,
                "input": state_text,
                "output": primitive_output,
                "valid": judge_result["valid"],
                "validation_reason": judge_result["reason"],
                "corrected_output" : judge_result["corrected_output"]
            })

            feedback_entries.append({
                "primitive_id": primitive_id,
                "input": state_text,
                "output": primitive_output,
                "valid": judge_result["valid"],
                "validation_reason": judge_result["reason"],
                "corrected_output": judge_result["corrected_output"]
            })

            # Update state for next primitive
            state_text = primitive_output
        
        except Exception as e:
            print(f"Error applying primitive {primitive_id}: {e}")
            return None, steps
    
    return state_text, steps , feedback_entries




def llm_validate_and_correct(judge_model, primitive_name, description, input_text, output_text):
    """
    Validate a primitive output and optionally correct it in a single LLM call.

    Returns:
        dict: {
            "valid": True/False/None,
            "reason": "short explanation",
            "corrected_output": <corrected output if invalid, or original output if valid>
        }
    """
    prompt = f"""
Primitive Name: {primitive_name}
Description: {description}
Input: {input_text}
Output: {output_text}

Question: Is the output correct? If not, provide the corrected output.

Respond with JSON:
{{
    "valid": true/false,
    "reason": "short explanation",
    "corrected_output": "..."
}}
"""
    response_text = judge_model.generate(prompt, max_new_tokens=128)
    try:
        result = json.loads(response_text)
        # Ensure corrected_output exists
        if "corrected_output" not in result or not result["corrected_output"]:
            result["corrected_output"] = output_text
        return result
    except Exception:
        return {
            "valid": None,
            "reason": "LLM response unparsable",
            "corrected_output": output_text
        }
