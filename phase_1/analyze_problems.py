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
Analyze problems and create reasoning plans in STRICT JSON format.

Your objectives:
1. Identify the type and domain of the problem.
2. Select 2–4 reasoning modules (from the list provided) that best suit this problem.
3. Identify appropriate methods or mental operations to use.
4.tags: include **short keywords describing problem characteristics or type**, e.g., "linear-equation", "counting", "geometry", "probability", "algebra". Avoid long sentences.
5. Generate decomposition strategies — short statements describing *how* to simplify reasoning.
6. Create a conceptual decomposition plan — a list of subgoals describing reasoning flow.
7.tags: include **short keywords describing problem characteristics or type**, e.g., "linear-equation", "counting", "geometry", "probability", "algebra". Avoid long sentences.

Always include the default reasoning module:
{default_modules}
Available reasoning modules (choose a few relevant ones):
{compact_modules}

Do not compute or give answers — focus only on reasoning structure.
'''

    user_prompt = f'''
        Problem: {question}

        Intermediate steps (if any): {steps}

        Format your response exactly as follows:

      <start>
      {{
      "problem_type": "",
      "domain": "",
      "sub_domain": "",
      "tags": [],
      "topics": [],
      "selected_reasoning_modules": [],
      "methods": [],
      "decomposition_strategies": ["Simplify structure", "Identify relations", "Break by variable type"],
      "decomposition_plan": [
        {{"goal": "", "description": ""}},
        {{"goal": "", "description": ""}}
      ]
      }}
      <end>

      STRICT RULES — fill these fields as instructed:
      1. problem_type (one concise label describing the cognitive task)
        - PURPOSE: describe *what kind of reasoning* the problem requires.
        - CHOOSE from (preferred) or use a short hyphenated phrase:
          [word_problem, equation_solving, counting, comparison, proof, algebraic_manipulation,
          simplification, inference, optimization, pattern_identification, unit_conversion,
          geometry_construction, probability_calculation, inverse_operation]
        - EXAMPLE: "word_problem" or "equation_solving".
        - DO NOT use generic values like "Math" or "Misc".

      2. domain (broad academic subject area)
        - PURPOSE: the major subject/discipline the problem belongs to.
        - CHOOSE one from:
          [Arithmetic, Algebra, Geometry, Number Theory, Probability, Combinatorics,
          Calculus, Linear Algebra, Statistics, Logic]
        - EXAMPLE: "Arithmetic".
        - Keep it one word or short phrase (e.g., "Linear Algebra").

      3. sub_domain (narrower, problem-specific area inside domain)
        - PURPOSE: a more specific tag inside the domain (helps primitive matching).
        - EXAMPLES: "money_word_problem", "linear_equation", "permutation_combination", "area_calculation".

      4. tags (short problem-characteristic keywords)
        - PURPOSE: quick machine-friendly keywords for filtering & matching primitives.
        - FORMAT: short, hyphenated or single-word tokens. Avoid sentences.
        - EXAMPLES: ["two-step", "addition-subtraction", "word-problem", "money", "linear-equation"].

      5. topics (curricular topics / learning objectives)
        - PURPOSE: curriculum-level topics the problem maps to.
        - EXAMPLES: ["addition & subtraction", "two-step word problems", "fractions", "probability basics"].

      ADDITIONAL RULES:
      - Fill every field. Use empty lists only if genuinely none apply.
      - Output ONLY the JSON between <start> and <end>. No extra text, no filler.
      - selected_reasoning_modules: choose 2–4 modules (include default Step-By-Step).
      - methods: list conceptual techniques (e.g., "inverse-solving", "transaction-modeling", "isolation").
      - decomposition_strategies: short phrases describing *how* you will simplify / chunk reasoning.
      - decomposition_plan: two or more conceptual subgoals (no numeric computation).

      Example (for clarity — do not echo this exact example back; use it as a model):

      <start>
      {{
      "problem_type": "word_problem",
      "domain": "Arithmetic",
      "sub_domain": "money_word_problem",
      "tags": ["two-step","addition-subtraction","money","word-problem"],
      "topics": ["addition & subtraction","two-step word problems"],
      "selected_reasoning_modules": ["Step-by-Step Reasoning","Simplification","Decomposition"],
      "methods": ["transaction-modeling","inverse-solving"],
      "decomposition_strategies": ["Model transactions sequentially","Isolate unknown by reversing operations"],
      "decomposition_plan": [
        {"goal": "Model final amount", "description": "Express final money as initial - spent + received."},
        {"goal": "Solve for initial amount", "description": "Rearrange transaction model to isolate initial amount."}
      ]
      }}
      <end>

    
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

      RULES:
      1. Your response must start with <start> and end with <end>
      2.Each subtask should be a single conceptual reasoning instruction.
      3.Use verbs like "identify", "compare", "estimate", "reason about", "infer".
      4. Nothing outside these markers - no greetings, no explanations
      5. JSON must be valid and parseable
      6. Focus on reasoning structure, not solutions

      '''
    
    complexity_estimate_2 = len(tokenizer(system_prompt_2 + user_prompt_2)["input_ids"])
    dynamic_max_tokens_2 = min(4096, max(400, 2 * complexity_estimate_2))


    phase2_output = generate_text(model, tokenizer, system_prompt_2, user_prompt_2, dynamic_max_tokens=dynamic_max_tokens_2)
        
    # merge results
    return {
        **phase1_output,
        "subtasks": phase2_output.get("subtasks", [])
    }
