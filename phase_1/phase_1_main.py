# phase_1_main.py

import json
from .preprocess import preprocess_problem
from .analyze_problems import analyze_problem


def run_phase1(model, tokenizer,problem=None, dataset_name="SVAMP"):
    """
    Run Phase 1: Preprocess and Analyze one problem.
    
    Args:
        problem (dict): Raw problem from dataset.
        dataset_name (str): Name of dataset ("SVAMP", "ASDiv", "GSM8K").
    
    Returns:
        tuple: (processed_json, analysis_json)
    """

    '''  Phase 1: Problem Analysis and Primitive Generation  '''

    # Step 1 - Data Preprocessing
    processed = preprocess_problem(problem, dataset_name=dataset_name)
    # print("\n=== Processed Problem (JSON) ===")
    # print(json.dumps(processed, indent=2))

    # Step 2 - Analyze the Problem
    analysis = analyze_problem(model, tokenizer ,processed)
    # print("\n=== Analysis (JSON) ===")
    # print(json.dumps(analysis, indent=2))

    return processed, analysis
