# analyze_problems.py

from typing import Dict, Any

from ..model_config import generate_text
from ..config import parse_raw_op_with_markers , Retries


# analyze_problems.py

from typing import Dict, Any
from ..model_config import generate_text
from ..config import parse_raw_op_with_markers, Retries


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



def analyze_and_decompose(model, tokenizer, problem_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1–2 unified analysis for reasoning and decomposition.
    1. Select reasoning modules (meta-analysis)
    2. Decompose problem into subtasks using selected modules
    """

    question = problem_entry.get("question", "")
    steps = problem_entry.get("intermediate_steps", "")

    # === Phase 1: Select & Analyze ===
    system_prompt = (
        "You are a meta-reasoning analyst inspired by the SELF-DISCOVER framework. "
        "Your goal is to analyze the given math problem, select reasoning modules that best fit the task, "
        "and design a conceptual decomposition plan. The default reasoning module (ID: 38) "
        "is always included. From the provided list of reasoning modules, select 2–4 additional modules "
        "that would most help solve the problem. "
        "Do not compute the answer — focus only on reasoning structure.\n\n"
        "Reasoning Modules JSON:\n"
        f"{reasoning_modules}"
    )

    user_prompt = f"""
        Problem: {question}

        Intermediate steps (if any): {steps}

        Format your response like this (between <start> and <end> markers):
        <start>
        {{
          "problem_type": "...",
          "domain": "...",
          "selected_reasoning_modules": ["...","..."],
          "methods": ["..."],
          "tags": ["..."],
          "analysis_summary": "...",
          "decomposition_plan": [
            {{"goal": "","description": ""}},
            {{"goal": "","description": ""}}
          ]
        }}
        <end>

        Guidelines:
        - Identify 2–4 reasoning modules most relevant to solving this task.
        - In 'analysis_summary', explain why those modules apply.
        - In 'decomposition_plan', outline conceptual subgoals (not computations).
        - Avoid algebraic or numeric steps; focus on reasoning structure.
        - Output one valid JSON object enclosed between <start> and <end> with no extra text.
        """

    # === dynamic token allocation ===
    complexity_estimate = len(tokenizer(user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(400, 2 * complexity_estimate))

    phase1_output = generate_text(model, tokenizer, system_prompt, user_prompt, max_tokens=dynamic_max_tokens)
    
    # === Phase 2: Decompose into Subtasks ===
    reasoning_modules = phase1_output.get("selected_reasoning_modules", [])
    decomposition_plan = phase1_output.get("decomposition_plan", [])

    system_prompt_2 = (
        "You are a structured reasoning planner. "
        "Using the reasoning modules and decomposition plan provided, "
        "generate clear subtasks that represent conceptual reasoning actions. "
        "Each subtask should directly reflect how those modules would be applied. "
        "Do not perform any calculation or algebraic manipulation."
    )

    user_prompt_2 = f"""
        Problem: {question}

        Selected reasoning modules: {reasoning_modules}
        Initial decomposition plan: {decomposition_plan}

        Format your response strictly as JSON between <start> and <end>:
        <start>
        {{
          "subtasks":[
            {{"step":1,"instruction":""}},
            {{"step":2,"instruction":""}},
            {{"step":3,"instruction":""}}
          ]
        }}
        <end>
        """

    phase2_output = generate_text(model, tokenizer, system_prompt_2, user_prompt_2, max_tokens=600)
        
    # merge results
    return {
        **phase1_output,
        "subtasks": phase2_output.get("subtasks", [])
    }
