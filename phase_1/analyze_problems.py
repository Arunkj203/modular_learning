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
    steps = problem_entry.get("intermediate_steps", "")

    # === Phase 1: Select & Analyze ===
    system_prompt = f'''You are a meta-reasoning architect.
Your task is to analyze a given problem and outline how a reasoning system
should think about it before attempting to solve it.

Your objectives:
1. Identify the type and domain of the problem.
2. Select 2–4 reasoning modules (from the list provided) that best suit this problem.
3. Identify appropriate methods or mental operations to use.
4. Generate decomposition strategies — short statements describing *how* to simplify reasoning.
5. Create a conceptual decomposition plan — a list of subgoals describing reasoning flow.

Always include the default reasoning module:
{default_modules}
Available reasoning modules (choose a few relevant ones):
{compact_modules}

Do not compute or give answers — focus only on reasoning structure.
'''

    user_prompt = f'''
        Problem: {question}

        Intermediate steps (if any): {steps}

        Format your response *exactly* as follows:
        <start>
        {{
          "problem_type": "...",
          "domain": "...",
          "selected_reasoning_modules": ["...","..."],
          "methods": ["..."],
          "decomposition_strategies": ["Simplify structure","Identify relations","Break by variable type"],
          "decomposition_plan": [
            {{"goal": "","description": ""}},
            {{"goal": "","description": ""}}
          ]
        }}
        <end>

        Rules:
        - Choose only 2–4 reasoning modules (include the default automatically).
        - Decomposition strategies describe *how* you plan to reason efficiently.
        - Decomposition plan lists conceptual reasoning goals, not numeric steps.
        - Output only JSON between <start> and <end>, no extra text or explanation.
        '''

    # === dynamic token allocation ===
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(400, 2 * complexity_estimate))

    phase1_output = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=dynamic_max_tokens)
    
    # === Phase 2: Decompose into Subtasks ===
    selected_modules = phase1_output.get("selected_reasoning_modules", [])
    decomposition_plan = phase1_output.get("decomposition_plan", [])

    system_prompt_2 = (
        "You are a structured reasoning planner. "
        "Using the reasoning modules and decomposition plan provided, "
        "generate clear subtasks that represent conceptual reasoning actions. "
        "Each subtask should directly reflect how those modules would be applied. "
        "Do not perform any calculation or algebraic manipulation."
    )

    user_prompt_2 = f'''
        Problem: {question}

        Selected reasoning modules: {selected_modules}
        Initial decomposition plan: {decomposition_plan}

        **Format your response strictly as JSON between <start> and <end>**:
        <start>
        {{
          "subtasks":[
            {{"step":1,"instruction":""}},
            {{"step":2,"instruction":""}},
            {{"step":3,"instruction":""}}
          ]
        }}
        <end>

      Guidelines:
      - Each subtask should be a single conceptual reasoning instruction.
      - Use verbs like "identify", "compare", "estimate", "reason about", "infer".
      - Do not restate the full problem or add commentary.
      - Output only the JSON enclosed between <start> and <end>.


      '''
    
    complexity_estimate_2 = len(tokenizer(system_prompt_2 + user_prompt_2)["input_ids"])
    dynamic_max_tokens_2 = min(4096, max(400, 2 * complexity_estimate_2))


    phase2_output = generate_text(model, tokenizer, system_prompt_2, user_prompt_2, dynamic_max_tokens=dynamic_max_tokens_2)
        
    # merge results
    return {
        **phase1_output,
        "subtasks": phase2_output.get("subtasks", [])
    }
