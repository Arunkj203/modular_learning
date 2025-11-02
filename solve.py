
from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3

from .phase_2.generate_primitive import add_primitive

from . import config as mem
from datasets import load_dataset

import os , json , re
from typing import Dict, Any, List


def generate_phase3_execution(phase2_file: str, model, tokenizer, output_dir="Dataset", use_lora=False):
    """
    Generate Phase 4 execution data from Phase 2 primitive sequences.

    Each entry in the Phase 2 file should contain:
        {
            "question": str,
            "phase2_reasoning": [list of primitives]
        }

    This function will execute the primitive sequence step-by-step using Phase 4
    and store the resulting state transitions for dataset preparation.
    """

    full_path = os.path.join(mem.Base_dir_path, output_dir)
    input_file = os.path.join(full_path, phase2_file)
    print(f"Loading Phase 2 file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output4_file = os.path.join(full_path, phase2_file.replace("phase2_execution", "phase4_execution"))
    print(f"Phase 4 logs will be saved to: {output4_file}")

    mem.load_memory()
    all_phase4_data = []
    errors = 0
    max_errors = int(0.3 * len(data))

    for idx, entry in enumerate(data[:2]):  # limit for testing
        question = entry.get("question", "")
        primitive_sequence = entry.get("phase2_reasoning", [])

        print(f"\n================== Problem {idx+1}/{len(data)} ==================")
        print(f"Executing {len(primitive_sequence)} primitives...")

        # try:
        final_state, steps, feedback = run_phase3(
            model,
            tokenizer,
            primitive_sequence,
            problem_text=question,
            use_lora=use_lora
        )

        # Prepare structured record for dataset training
        all_phase4_data.append({
            "id": idx,
            "question": question,
            "primitive_sequence": primitive_sequence,
            "execution_trace": steps,        # step-by-step transformation logs
            "final_state": final_state,
            "feedback": feedback
        })

        print(f"  ✅ Completed problem {idx+1} ({len(steps)} steps)")

        # except Exception as e:
        #     errors += 1
        #     print(f"  [ERROR] Problem {idx+1} failed: {e}")
        #     if errors >= max_errors:
        #         print(f"\n[ABORT] Too many errors ({errors}). Stopping early.\n")
        #         break

    # Save the dataset
    with open(output4_file, "w", encoding="utf-8") as f:
        json.dump(all_phase4_data, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 4 execution dataset saved to: {output4_file}")


def generate_phase2_execution(phase1_file: str, model, tokenizer, output_dir="Dataset"):
    """
    Generate Phase 2 reasoning outputs from Phase 1 analyses.
    """

    full_path = os.path.join(mem.Base_dir_path, output_dir)
    output_file = os.path.join(full_path,phase1_file)
    print(f"Log saving in file:{output_file}")

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    output2_file = os.path.join(full_path, phase1_file.replace("phase1_analysis", "phase2_execution[testing]"))

    # l = len(data)
    l = 3
    batch_size = max(1,l // 10)
    all_results = []

    no_errors = 0
    max_errors = int(0.3 * l)

    mem.load_memory()

    for batch_start in range(0, l , batch_size):
        batch = data[batch_start:batch_start + batch_size]
        batch_new_primitives = []

        print(f"\nProcessing batch {batch_start // batch_size + 1}...")

        for entry in batch:
            q = entry["question"]
            analysis = entry["phase1_analysis"]

            try:

                print(f"  Executing reasoning for problem {entry['id']}...")

                primitive_sequence, new_prims = run_phase2(model, tokenizer, q, analysis)
                batch_new_primitives.extend(new_prims)

                all_results.append({
                    "question": q,
                    "phase2_reasoning": primitive_sequence
                })
            except Exception as e:
                no_errors += 1
                print(f"  [ERROR] Problem {entry['id']} failed: {e}")
        

        # --- After each batch ---
        if batch_new_primitives:
            print(f"  Adding {len(batch_new_primitives)} new primitives from this batch...")
            for prim in batch_new_primitives:
                add_primitive(prim)

            print(f"  Library updated. Total primitives now: {len(mem.primitive_metadata)}")
        
        if no_errors >= max_errors:
            print(f"\n[ABORT] Too many errors ({no_errors}). Stopping early in batch {batch_start}.\n")
            break

    # Save all results at the end
    with open(output2_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    mem.save_memory()
    print(f"\nPhase 2 reasoning saved to {output2_file}")


def generate_phase1_analysis(dataset_name: str, mode: str, model, tokenizer, output_dir="Dataset"):
    """
    Generate Phase 1 analysis for all problems in the dataset and store results in a JSON file.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'svamp').
        mode (str): Dataset split (e.g., 'train', 'test', 'validation').
        model: Loaded model for Phase 1.
        tokenizer: Tokenizer corresponding to the model.
        output_dir (str): Directory to save JSON analysis.

    Returns:
        str: Path to the saved JSON file.
    """
    # Ensure log directory exists
    full_path = os.path.join(mem.Base_dir_path, output_dir)
    os.makedirs(full_path, exist_ok=True)
    output_file = os.path.join(full_path, f"{dataset_name}_{mode}_phase1_analysis.json")
    print(f"Log saving in file:{output_file}")


    dataset = list(load_dataset(mem.dataset_path[dataset_name])[mode]) 
    all_analysis: List[Dict[str, Any]] = []

    for idx, problem in enumerate(dataset):
        print(f"Analyzing problem {idx+1}/{len(dataset)}")
        # try:

        processed, analysis = run_phase1(model, tokenizer, problem, dataset_name=dataset_name)
        entry = {
            "id": idx,
            "question": processed.get("question", ""),
            "ground_truth": normalize_answer(processed.get("answer", "")),
            "phase1_analysis": analysis
        }
        all_analysis.append(entry)
        
        # except Exception as e:
        #     print(f"[ERROR] Problem {idx+1} failed: {e}")
        #     all_analysis.append({
        #         "id": idx,
        #         "question": problem.get("question", ""),
        #         "error": str(e)
        #     })


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_analysis, f, indent=2, ensure_ascii=False)

    print(f"Phase 1 analysis saved to {output_file}")
    # return output_file


def solve(dataset_name, mode, mode_text, model, tokenizer, log_dir="logs"):

    correct, total = 0, 0
    primitive_logs = []
    all_feedback = []  # Collect feedback for all problems

    max_errors = 4
    errors = 0 


    use_lora = False
    dataset = load_dataset(mem.dataset_path[dataset_name])
    mem.load_memory()

    # Ensure log directory exists
    full_path = os.path.join(mem.Base_dir_path, log_dir)
    os.makedirs(full_path, exist_ok=True)
    log_file = os.path.join(full_path, f"{dataset_name}_{mode}-02.11__10_pbs.txt")
    print(f"Log saving in file:{log_file}")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== {mode_text} on {dataset_name} ===\n\n")


        for idx, problem in enumerate(list(dataset[mode])[15:25]):  # limit for testing
            print(f"====================== Problem {idx+1} ======================")

            f.write(f"\n====================== Problem {idx+1} ======================\n")
            
            try:

                # Phase 1: Problem Analysis
                print(f"\nPhase 1 - Analysing...\n")

                processed, analysis = run_phase1(model, tokenizer, problem, dataset_name=dataset_name)
                gt = normalize_answer(processed["answer"])
                f.write(f"\nQuestion:\n{processed['question']}\n")
                f.write(f"\nGround Truth Answer:\n{gt}\n")
                f.write(f"\nPhase 1 - Analysis:\n{analysis}\n")
                

                # Phase 2: Primitive Generation
                print(f"\nPhase 2 - Primitive Sequence Generating...\n")

                primitive_sequence, new_primitives_to_train = run_phase2(model, tokenizer, processed["question"], analysis)
                f.write("\nPhase 2 - Primitive Sequence:\n")
                f.write(f"\n{len(new_primitives_to_train)} new primitves generated out of {len(primitive_sequence)}\n")

                for prim in primitive_sequence:
                    f.write(f"  ID: {prim['id']} \n, prim: {prim} \n")

                print(f"\n{len(new_primitives_to_train)} new primitves generated out of {len(primitive_sequence)}\n")
                
                
                # Phase 4: Problem Solving
                print("\nPhase 3 -  Solving...\n")

                solution, steps, feedback_entries = run_phase3(
                    model, tokenizer, primitive_sequence, problem_text=processed["question"]
                )
                pred = normalize_answer(solution)

                f.write("\nPhase 3 - Execution Steps:\n")
                for step in steps:
                    f.write(f"  Primitive name :{step[2]}:\n")
                    f.write(f"    Output: {step[0]}\n")
                
                
                f.write(f"\nFinal Solution:  {solution}\nNormalized solution:{pred}\n")
                
                print(f"\n====================== Problem {idx+1} Solved ======================\n")

                # Collect all feedback
                # all_feedback.extend(feedback_entries) 
                # Changes need to be made in phase 4 (return feedback entries)


                # Track accuracy
                
                if pred == gt:
                    correct += 1
                total += 1

            except Exception as e:
                errors += 1
                f.write(f"\n[ERROR] Problem {idx+1} failed: {e}\n")
                
                print(f"\n[ERROR] Problem {idx+1} failed: {e}\n")

                if errors >= max_errors:
                    f.write(f"\n[ABORT] Too many errors ({errors}). Stopping early.\n")
                    print(f"\n[ABORT] Too many errors ({errors}). Stopping early.\n")
                    break

        # Write accuracy at the end
        acc = correct / total if total > 0 else 0
        f.write(f"\n\n=== Accuracy: {acc:.2f} ({correct}/{total}) ===\n")

        # Save memory
        mem.save_memory()


    return acc,all_feedback



def normalize_answer(text: str):
    """
    Normalize a model's final output for comparison with ground truth.
    Works across math, logic, and general text reasoning datasets.
    """

    if text is None:
        return None

    text = str(text).strip()

    # --- 1. JSON safety: if the model wrapped its answer in JSON ---
    try:
        obj = json.loads(text)
        # flatten if it's a dict with one field like {"answer": 4}
        if isinstance(obj, dict) and len(obj) == 1:
            text = list(obj.values())[0]
        elif isinstance(obj, list) and len(obj) == 1:
            text = obj[0]
    except Exception:
        pass

    # --- 2. Remove common prefixes/suffixes ---
    text = re.sub(r'(?i)\b(answer|final answer|result|output|solution)\b[:\-\s]*', '', text)
    text = text.strip().replace("=", "").strip()

    # --- 3. Extract numbers if present ---
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if numbers:
        # Pick the last number as the most likely final value
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    # --- 4. If it's symbolic math (like 'x=4' or '104/26=4') ---
    if re.search(r'[\+\-\*/=]', text):
        try:
            # Evaluate the last part after '=' if any
            if '=' in text:
                expr = text.split('=')[-1].strip()
            else:
                expr = text
            result = eval(expr, {"__builtins__": None}, {})
            return float(result)
        except Exception:
            pass

    # --- 5. Fallback: textual normalization ---
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text




