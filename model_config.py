
# model_config.py
from transformers import AutoTokenizer, AutoModelForCausalLM ,StoppingCriteria, StoppingCriteriaList 

import torch, gc, re, json , requests , os
from transformers import GenerationConfig , BitsAndBytesConfig

# Optional: load .env
try:
    from dotenv import load_dotenv 
    load_dotenv()
except Exception:
    pass




# -----------------------------
# Config: model selection
# -----------------------------


# OLD: BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

# BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct" # For phase 1 - Analysis

# BASE_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct" # For phase 2 - Synthesis

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # For phase 3 - Execution



# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # New Model

# BASE_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # Test Model
OUTPUT_DIR = "./results/lora_adapters"



# DEVICE = "cuda"  # full GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Constants
Retries = 3


# -----------------------------
# Singleton loader
# -----------------------------

def get_model_and_tokenizer():
    print(f"Loading tokenizer for {BASE_MODEL}...")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     BASE_MODEL,
    #     use_fast=True,
    #     local_files_only=True  # avoid re-downloading if already cached
    # )

    tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HUGGINGFACEHUB_API_TOKEN,
    use_fast=True,
    local_files_only=False,  # allow fetching updated tokenizer if missing
    trust_remote_code=True   # needed for new chat template logic
)


    print(f"Loading model {BASE_MODEL} in 4-bit mode on GPU...")

    # # 4-bit quantization config (QLoRA style)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     low_cpu_mem_usage=True,
    #     local_files_only=True
    # )

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,                # <-- switch from 4-bit to 8-bit
        llm_int8_threshold=6.0,           # auto-determine outlier handling
        llm_int8_has_fp16_weight=False,   # recommended for inference
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=HUGGINGFACEHUB_API_TOKEN
    )

    # optional: faster generation tweaks
    model.eval()
    model.config.use_cache = True
    model.config.pretraining_tp = 1
    
    return model, tokenizer

# -----------------------------
# Helper function for generation
# -----------------------------

# The StopOnToken class is acceptable and kept as is, as it's a robust custom implementation for delimiters.
class StopOnToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_token):

        self.tokenizer = tokenizer
        self.stop_token = stop_token
        self.stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        # Check if the last tokens match our stop token
        if input_ids.shape[1] < len(self.stop_token_ids):
            return False
        
        # Get the recent tokens that could match our stop token
        recent_tokens = input_ids[0, -len(self.stop_token_ids):].tolist()
        return recent_tokens == self.stop_token_ids

def generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=200, Retries=3, DEVICE="cuda"):
    import torch, gc, re, json
    from transformers import GenerationConfig, StoppingCriteriaList

    # Pre-clear memory to avoid fragmentation from previous runs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use chat template for Qwen/DeepSeek-style models
    prompt_string = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    raw = None
    last_error = None

    
    stop_criteria = StoppingCriteriaList([StopOnToken(tokenizer, "<<END>>")])

    for attempt in range(Retries):
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt_string, return_tensors="pt", add_special_tokens=True).to(DEVICE)
            prompt_len = inputs["input_ids"].shape[-1]

            max_tokens = min(4096, dynamic_max_tokens * (2 ** attempt))
            gen_cfg = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Clear memory again before the heaviest call
            torch.cuda.empty_cache()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    stopping_criteria=stop_criteria
                )

            generated_tokens = outputs[0][prompt_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

            
            json_op = extract_and_clean_json(generated_text)
            # Cleanup memory before returning
            del outputs, inputs
            torch.cuda.empty_cache()
            gc.collect()

            return json_op

        except Exception as e:
            last_error = e
            debug_raw = generated_text if 'generated_text' in locals() else "<no raw output>"
            print(f"[WARN] Attempt {attempt+1} failed: {type(e).__name__}: {e}\nRaw output:\n{debug_raw}\n")
            # print(f"[WARN] Attempt {attempt+1} failed: {type(e).__name__}: {e}\n")

            # Free memory after each failed attempt
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

    # Final cleanup before exiting
    torch.cuda.empty_cache()
    gc.collect()
    # raise RuntimeError(f"Failed after {Retries} attempts.\nLast error: {last_error}\nRaw: {raw}")
    raise RuntimeError(f"Failed after {Retries} attempts.\nLast error: {last_error}")

def extract_and_clean_json(generated_text: str):
    """
    Extract valid JSON from model output, removing artifacts and fixing quotes.
    Handles:
      - <<START>> ... <<END>> blocks
      - <|eot_id|> tokens and other artifacts
      - unescaped double quotes inside values
      - mixed single/double quotes and trailing commas
    """

    # --- 0. Cut off any prefix up to <|eot_id|> ---
    generated_text = re.split(r"<\|eot_id\|>", generated_text, 1)[0]
    generated_text = re.sub(r"<\|.*?\|>", "", generated_text).strip()

    # --- 1. Extract JSON substring ---
    match = re.search(r"<<START>>\s*([\s\S]*?)\s*<<END>>", generated_text, flags=re.S)
    if match:
        json_text = match.group(1).strip()
    else:
        m = re.search(r"([\{\[][\s\S]*[\}\]])", generated_text, flags=re.S)
        if m:
            json_text = m.group(1).strip()
        else:
            raise ValueError("Could not find JSON object or <<START>>...<<END>> delimiters.")

   # --- 2. Basic sanitization ---
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)   # remove trailing commas
    # json_text = re.sub(r"\n+", " ", json_text)             # flatten newlines
    json_text = json_text.strip()

     # --- 3. Convert single quotes ONLY around keys/values, not inside sentences ---
    json_text = re.sub(r"([{,]\s*)'([^']+?)'\s*:", r'\1"\2":', json_text)  # key fix
    json_text = re.sub(r":\s*'([^']*?)'", r': "\1"', json_text)   # value fix


    # --- 4. Escape unescaped inner double quotes ---
    def escape_inner_quotes(s):
        def repl(match):
            inner = match.group(1)
            fixed = inner.replace('"', '\\"')
            return f'"{fixed}"'
        return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', repl, s)
    json_text = escape_inner_quotes(json_text)

    # --- 5. Attempt parse with optional brace fix ---
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # maybe missing a closing brace
        if not json_text.strip().endswith("}"):
            json_text = json_text.strip() + "}"
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print("[ERROR] JSON still invalid after cleanup.")
            print("Cleaned text preview:", json_text[:500])
            raise e

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def call_openrouter(system_prompt: str, user_prompt: str, model="meta-llama/llama-3.3-70b-instruct:free", max_tokens=1000, temperature=0.0):
    """
    Call LLaMA-3.3-70B Instruct via OpenRouter API.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    return data["choices"][0]["message"]["content"].strip()
