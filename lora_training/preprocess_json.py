import json,os
import random
from datasets import Dataset, DatasetDict

from ..config import Base_dir_path

# ----------------------------
# Common Helpers
# ----------------------------

def load_json_records(path):
    """Load a JSON file and always return a list of records."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

    # Convert dict formats to list
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]

    return data[:10] 


def split_dataset(examples, val_split, seed):
    """Shuffle and split into train + val datasets."""
    random.seed(seed)
    random.shuffle(examples)

    n_val = int(len(examples) * val_split)
    val = examples[:n_val]
    train = examples[n_val:]

    return DatasetDict({
        "train": Dataset.from_list(train),
        "val": Dataset.from_list(val)
    })


# ----------------------------
# Phase 1
# ----------------------------

def preprocess_phase1_dataset(json_files, val_split=0.1, seed=42):
    examples = []

    for path in json_files:

        full_path = os.path.join(Base_dir_path,path)
        for item in load_json_records(full_path):
            q = item.get("question", "").strip()
            p1 = item.get("phase1_analysis", {})

            problem_type = p1.get("problem_type", "Unknown Type")
            topics = ", ".join(p1.get("topics", []))
            modules = ", ".join(m.get("name", "") for m in p1.get("selected_modules", []))

            text = (
                f"Question: {q}\n"
                f"Output: Problem Type: {problem_type}; "
                f"Topics: {topics}; Modules: {modules}"
            )
            examples.append({"text": text})

    print(f"[Phase1] Total={len(examples)}")
    return split_dataset(examples, val_split, seed)


# ----------------------------
# Phase 2
# ----------------------------

def preprocess_phase2_dataset(json_files, val_split=0.1, seed=42):
    examples = []

    for path in json_files:
        full_path = os.path.join(Base_dir_path,path)
        for item in load_json_records(full_path):
            q = item.get("question", "").strip()
            steps = item.get("phase2_reasoning", [])

            # Build full reasoning output with id + name + status
            reasoning = "\n".join(
                f"{s.get('step','?')}. [{s.get('id','')}] "
                f"{s.get('name','').strip()} "
                f"(Status: {s.get('status','').strip()})"
                for s in steps
            )


            text = f"Question: {q}\nOutput:\n{reasoning}"
            examples.append({"text": text})

    print(f"[Phase2] Total={len(examples)}")
    return split_dataset(examples, val_split, seed)


# ----------------------------
# Phase 3
# ----------------------------

def preprocess_phase3_dataset(json_files, val_split=0.1, seed=42):
    examples = []

    for path in json_files:
        full_path = os.path.join(Base_dir_path,path)
        for item in load_json_records(full_path):
            q = item.get("question", "").strip()

            # Primitive sequence
            prim_seq = "\n".join(
                f"{s.get('step', '?')}. {s.get('name', '').strip()}"
                for s in item.get("primitive_sequence", [])
            )

            # Execution trace
            exec_lines = []
            for idx, trace in enumerate(item.get("execution_trace", []), start=1):
                try:
                    info = trace[0]
                    name = info.get("primitive_applied", {}).get("name", "Unknown")
                    result = info.get("result", "").strip()
                    exec_lines.append(f"Step {idx}: {name} -> {result}")
                except:
                    continue

            exec_seq = "\n".join(exec_lines)

            final_state = item.get("final_state", "").strip()

            text = (
                f"Question: {q}\nOutput:\n"
                f"Primitive Sequence:\n{prim_seq}\n\n"
                f"Execution Trace:\n{exec_seq}\n\n"
                f"Final State: {final_state}"
            )

            examples.append({"text": text})

    print(f"[Phase3] Total={len(examples)}")
    return split_dataset(examples, val_split, seed)
