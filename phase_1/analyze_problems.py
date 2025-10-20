# analyze_problems.py

from typing import Dict, Any
from ..model_config import generate_text
import json

reasoning_modules = {
  "default_reasoning_module": [
      {
    "name": "Step-by-Step Reasoning",
    "description": "Let’s think step by step to ensure logical and systematic reasoning."
  },
  {
      "name": "Core Problem Identification",
      "description": "Clarify the exact question being asked and isolate the essential variables or goals."
    },
    {
      "name": "Stepwise Planning",
      "description": "Lay out an explicit sequence of steps or operations to follow in solving the problem."
    }
    ],
  "available_reasoning_modules": [
    # --- Tier 1: Foundational procedural modules (used for almost all problems)
    {
      "name": "Simplification",
      "description": "Rephrase or reduce the problem into an easier equivalent form while preserving meaning."
    },
    {
      "name": "Decomposition",
      "description": "Break down the problem into smaller, sequential or parallel parts that are easier to solve individually."
    },

    # --- Tier 2: Analytical and logical enhancement
    {
      "name": "Critical Thinking",
      "description": "Evaluate the reasoning process, test assumptions, and verify the validity of each logical step."
    },
    {
      "name": "Cause Analysis",
      "description": "Identify underlying relationships, causes, or dependencies that explain the situation or outcome."
    },
    {
      "name": "Assumption Analysis",
      "description": "Recognize and test implicit assumptions that influence reasoning or interpretation of the problem."
    },

    # --- Tier 3: Reflective and adaptive reasoning
    {
      "name": "Progress Evaluation",
      "description": "Monitor intermediate results, check for consistency, and verify whether the solution is converging toward correctness."
    },
    {
      "name": "Reflective Thinking",
      "description": "Review previous steps or mistakes to improve the reasoning process and future performance."
    },

    # --- Tier 4: Divergent or higher-order reasoning
    {
      "name": "Creative Thinking",
      "description": "Explore unconventional strategies, analogies, or representations to approach the problem from new perspectives."
    },
    {
      "name": "Alternative Solution Generation",
      "description": "Formulate other possible solution paths or hypotheses beyond the initial approach."
    },
    {
      "name": "Solution Reframing",
      "description": "Re-express the problem or shift perspective when the current approach is unproductive or incomplete."
    },
    {
      "name": "Systems Thinking",
      "description": "View the problem as part of a larger interconnected system and analyze how changes in one part affect the rest."
    }
  ]
}

problem_types = [
  "Arithmetic Reasoning",          # word problems requiring multi-step arithmetic
  "Algebraic Reasoning",           # finding unknowns, solving equations
  "Comparative Reasoning",         # difference, ratio, percent comparison
  "Proportional Reasoning",        # scaling, unit conversion, fractions
  "Counting and Combinatorics",    # number of ways, arrangements
  "Measurement and Geometry",      # area, perimeter, time, speed-distance
  "Logical Deduction",             # indirect reasoning (e.g., if A has more than B...)
  "Data Interpretation"            # reading tables, interpreting given quantities
]

topics = [
  "Addition " , 
  "Subtraction",
  "Multiplication",
  "Division",
  "Fractions and Ratios",
  "Percentages",
  "Units and Conversions",
  "Time, Speed, and Distance",
  "Average and Mean",
  "Work and Rate Problems",
  "Simple Algebra",
  "Comparisons and Ordering",
  "Logical Conditions",
  "Word Parsing and Translation"
]

tags = [
  "multi-step",
  "single-step",
  "comparison",
  "causal",
  "proportional",
  "symbolic",
  "linguistic-parsing",
  "data-driven",
  "logical",
  "estimation",
  "contextual",
  "arithmetic-chain",
  "unit-consistency"
]


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

    system_prompt  = f"""
          You are an expert reasoning analyst specialized in arithmetic and logical word problems
          (SVAMP and GSM8K style). Your job is to analyze a math or logic problem and describe its
          structure, type, and relevant reasoning modules. You do NOT solve the problem.

          Follow these strict rules:

          1. Output ONLY a single JSON object, wrapped between the literal delimiters <<START>> and <<END>>.
          2. Do NOT add any explanation, reasoning, or commentary outside the JSON.
          3. Ensure the JSON is strictly valid: no comments, no extra commas, no markdown formatting.
          4. Choose only from the predefined categories below — do NOT invent new labels.

            • problem_type (one of): {', '.join(problem_types)}
            • topics (select two or more): {', '.join(topics)}
            • tags (select five or more): {', '.join(tags)}
            • selected_modules (choose from these):
              Default modules: {default_modules}
              Available modules: {compact_modules}

          Your role is purely analytical — describe the reasoning structure and thinking process categories.
          """


    user_prompt = f"""
            Problem:
            {question}

            Output schema:
            <<START>>
            {{
              "problem_type": "<choose one from predefined list>",
              "topics": ["<choose one or more from predefined list>"],
              "tags": ["<choose one or more from predefined list>"],
              "selected_modules": ["<choose one or more reasoning modules>"]
            }}
            <<END>>

            Now generate ONLY the JSON analysis within the delimiters.
            """

    # print("\n=== Phase 1: Problem Analysis Prompt ===")
    # === dynamic token allocation ===
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(400, 2 * complexity_estimate))

    phase1_output = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=dynamic_max_tokens)
    
    # print("\n=== Phase 1: Problem Analysis Output ===")
    return phase1_output  
   