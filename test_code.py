from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

import json
import os
from tqdm import tqdm

from .phase_1.phase_1_main import run_phase1
from .phase_2.phase_2_main import run_phase2
from .phase_3.phase_3_main import run_phase3

from .phase_2.generate_primitive import add_primitive, update_primitive_graph_from_sequence

from . import config as mem
from .solve import normalize_answer


# Optional: load .env
try:
    from dotenv import load_dotenv  
    load_dotenv()
except Exception:
    pass
    
HUGGINGFACEHUB_API_TOKEN_3B = os.getenv("HUGGINGFACEHUB_API_TOKEN_3B")

def get_model_and_tokenizer(BASE_MODEL):

    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HUGGINGFACEHUB_API_TOKEN_3B,
        use_fast=True,
        trust_remote_code=True
    )

    print(f"Loading full precision model {BASE_MODEL} on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,   # use float16 for efficiency
        device_map="auto",
        trust_remote_code=True,
        token=HUGGINGFACEHUB_API_TOKEN_3B
    )

    model.eval()
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    return model, tokenizer


def load_datasets():
    svamp = load_dataset("ChilleD/SVAMP", split="test")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded SVAMP test: {len(svamp)} samples")
    print(f"Loaded GSM8k test: {len(gsm8k)} samples")
    return svamp, gsm8k


def generate_phase1_analysis(dataset, model, tokenizer, output_dir="Base_L", dataset_name="svamp"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_test_phase1_analysis.json")
    
    results = []
    l = len(dataset)
    for i, problem in enumerate(dataset):
        
        print(f"Analyzing problem {i+1}/{l}")
        try:
            processed, analysis = run_phase1(model, tokenizer, problem, dataset_name=dataset_name)
            entry = {
                "id": i,
                "question": processed.get("question", ""),
                "ground_truth": normalize_answer(processed.get("answer", "")),
                "phase1_analysis": analysis
            }
            results.append(entry)
        except Exception as e:
            print(f"[ERROR] Problem {i+1} failed: {e}")
            results.append({
                "id": i,
                "question": problem.get("question", ""),
                "error": str(e)
            })

    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved Test-Phase 1 analysis for {dataset_name} at {output_file}")

if __name__ == "__main__":
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # or your local model path
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)
    
    print("Load dataset")
    svamp, gsm8k = load_datasets()
    
    generate_phase1_analysis(svamp, model, tokenizer, dataset_name="svamp")
    generate_phase1_analysis(gsm8k, model, tokenizer, dataset_name="gsm8k")

