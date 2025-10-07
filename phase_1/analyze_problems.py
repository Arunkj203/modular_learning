# analyze_problems.py

from typing import Dict, Any
from ..model_config import generate_text
import json

reasoning_modules = {
  "default_reasoning_module": {
    "name": "Step-by-Step Reasoning",
    "description": "Let’s think step by step to ensure logical and systematic reasoning."
  },
  "available_reasoning_modules": [
    {
      "name": "Simplification",
      "description": "How can I simplify the problem so that it is easier to solve?"
    },
    {
      "name": "Assumption Analysis",
      "description": "What are the key assumptions underlying this problem?"
    },
    {
      "name": "Decomposition",
      "description": "How can I break down this problem into smaller, more manageable parts?"
    },
    {
      "name": "Critical Thinking",
      "description": "Analyze the problem from different perspectives, question assumptions, and evaluate reasoning logically."
    },
    {
      "name": "Creative Thinking",
      "description": "Generate innovative or unconventional ideas to approach the problem from new angles."
    },
    {
      "name": "Systems Thinking",
      "description": "Consider the problem as part of a larger system and understand interdependencies among elements."
    },
    {
      "name": "Reflective Thinking",
      "description": "Re-examine reasoning steps, question personal biases, and learn from prior attempts or failures."
    },
    {
      "name": "Core Problem Identification",
      "description": "What is the core issue or goal that needs to be addressed?"
    },
    {
      "name": "Cause Analysis",
      "description": "What are the underlying factors or relationships that contribute to this problem?"
    },
    {
      "name": "Progress Evaluation",
      "description": "How can progress or success in solving the problem be measured or verified?"
    },
    {
      "name": "Alternative Solution Generation",
      "description": "Given the current best solution, can we hypothesize other possible approaches?"
    },
    {
      "name": "Solution Reframing",
      "description": "If the current best solution is wrong, what other perspectives or methods could work?"
    },
    {
      "name": "Stepwise Planning",
      "description": "Let’s make a clear step-by-step plan and implement it with explanation."
    }
  ]
}

default_modules = json.dumps(reasoning_modules['default_reasoning_module'], separators=(",", ":"), ensure_ascii=False)
compact_modules = json.dumps(reasoning_modules['available_reasoning_modules'], separators=(",", ":"), ensure_ascii=False)



def analyze_and_decompose(model, tokenizer, problem_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1–2 unified analysis for reasoning and decomposition.
    1. Select reasoning modules (meta-analysis)
    2. Decompose problem into subtasks using selected modules
    """

    question = problem_entry.get("question", "")
    # steps = problem_entry.get("intermediate_steps", "")

    system_prompt = f"""
    You are an expert mathematician and AI reasoning analyst. Your task is to analyze a given problem, determine its type, topics, tags, applicable methods, and decompose it into structured sub-tasks.

    **CRITICAL FORMATTING INSTRUCTIONS:**
    1. Your entire response MUST be a single, valid JSON object.
    2. Wrap the JSON output strictly between the literal strings <<START>> and <<END>>.
    3. DO NOT include any introductory text, concluding remarks, or any other prose outside of the JSON structure.
    4. Select reasoning modules from the following:
      - Default modules: {default_modules}
      - Available modules: {compact_modules}
    5. Divide the problem into numbered sub-tasks and explain which module(s) you would use for each sub-task.
    6. You do NOT need to provide the solution or calculations — only analysis.
    """

    user_prompt = f"""
    **Problem to Analyze:**
    {question}

    **REQUIRED JSON OUTPUT SCHEMA (ANALYSIS ONLY):**
    <<START>>
    {{
      "problem_type": "<categorize problem type>",
      "topics": ["<list relevant topics or domains>"],
      "tags": ["<list relevant tags or keywords>"],
      "selected_modules": ["<select relevant reasoning modules from default or available>"],
      "sub_tasks": [
        {{"task": "Describe sub-task 1 and associated module(s)"}},
        {{"task": "Describe sub-task 2 and associated module(s)"}},
        {{"task": "Describe sub-task 3 and associated module(s)"}}
      ]
    }}
    <<END>>

    **GENERATE THE ANALYSIS JSON NOW.**
    """

    
    # === dynamic token allocation ===
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(400, 2 * complexity_estimate))

    phase1_output = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=dynamic_max_tokens)
    
    return phase1_output
   