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
    system_prompt = f'''You are a JSON-only analysis machine. You output nothing but valid JSON between <start> and <end>.
                    STRICT RULES:
                    1. Your response MUST begin with: <start>
                    2. Your response MUST end with: <end> 
                    3. Between these markers: ONLY valid JSON, no other text
                    4. Violating this format is a critical failure

                    Analysis objectives:
                    - Classify problem type, domain, sub_domain
                    - Select 2-4 reasoning modules (include: {default_modules})
                    - Define methods and decomposition strategies
                    - Create conceptual reasoning plan

                    Available modules: {compact_modules}
                    '''

    user_prompt = f'''
            INPUT PROBLEM: {question}
            INTERMEDIATE STEPS: {steps}

            OUTPUT REQUIREMENT: You must output ONLY this exact format:

            <start>
            {{
              "problem_type": "choose_from:word_problem,equation_solving,counting,comparison,proof,algebraic_manipulation,simplification,inference,optimization,pattern_identification,unit_conversion,geometry_construction,probability_calculation,inverse_operation",
              "domain": "choose_from:Arithmetic,Algebra,Geometry,Number_Theory,Probability,Combinatorics,Calculus,Linear_Algebra,Statistics,Logic",
              "sub_domain": "specific_area_like_money_word_problem_or_linear_equation",
              "tags": ["keyword1", "keyword2", "keyword3"],
              "topics": ["topic1", "topic2"],
              "selected_reasoning_modules": ["Step-By-Step", "module2", "module3"],
              "methods": ["method1", "method2"],
              "decomposition_strategies": ["strategy1", "strategy2"],
              "decomposition_plan": [
                {{"goal": "first_conceptual_goal", "description": "what_this_accomplishes"}},
                {{"goal": "second_conceptual_goal", "description": "what_this_accomplishes"}}
              ]
            }}
            <end>

            FIELD GUIDELINES:
            - problem_type: One concise label from the provided list
            - domain: One broad subject area from provided list  
            - sub_domain: Specific area within domain (be precise)
            - tags: 2-5 machine-friendly keywords describing problem characteristics
            - topics: 1-3 curricular learning objectives
            - selected_reasoning_modules: 2-4 modules including "Step-By-Step"
            - methods: Conceptual techniques used in reasoning
            - decomposition_strategies: How to break down the problem
            - decomposition_plan: 2-4 conceptual reasoning goals (not computational steps)

            BEGIN OUTPUT NOW. REMEMBER: <start> first, <end> last, JSON only between them.
            '''
    
    # === dynamic token allocation ===
    complexity_estimate = len(tokenizer(system_prompt + user_prompt)["input_ids"])
    dynamic_max_tokens = min(4096, max(400, 2 * complexity_estimate))

    phase1_output = generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=dynamic_max_tokens)
    
    print("\nDecomposing...\n")
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
