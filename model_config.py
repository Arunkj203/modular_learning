
# model_config.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM ,StoppingCriteria, StoppingCriteriaList 
import os,requests,re, json

from transformers import GenerationConfig

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
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # New Model

# BASE_MODEL = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # Test Model
OUTPUT_DIR = "./results/lora_adapters"



# DEVICE = "cuda"  # full GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Constants
Retries = 1


# -----------------------------
# Singleton loader
# -----------------------------


def get_model_and_tokenizer():

    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,token=HUGGINGFACEHUB_API_TOKEN)

    print(f"Loading model {BASE_MODEL} on {DEVICE} (INT8)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.float16,
        token = HUGGINGFACEHUB_API_TOKEN
    )

    return model , tokenizer

# -----------------------------
# Helper function for generation
# -----------------------------


import torch
import re
import json
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList # Assuming these are imported

# The StopOnToken class is acceptable and kept as is, as it's a robust custom implementation for delimiters.
class StopOnToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_token):
        self.tokenizer = tokenizer
        self.stop_token = stop_token
        # Ensure we encode the stop token string
        self.stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_len = len(self.stop_token_ids)
        if input_ids.shape[1] < stop_len:
            return False
            
        # Check if the last tokens match the stop sequence
        recent_tokens = input_ids[0, -stop_len:].tolist()
        return recent_tokens == self.stop_token_ids

def generate_text(model, tokenizer, system_prompt, user_prompt, dynamic_max_tokens=200, Retries=3, DEVICE="cuda"): # Added Retries/DEVICE for completeness
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # ***CRITICAL FIX 1: Use apply_chat_template for chat-tuned models (Qwen/DeepSeek)***
    # This generates the correctly tokenized prompt string with roles.
    prompt_string = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    raw = None
    last_error = None
    
    # Tokenize the final prompt. Added attention_mask handling.
    inputs = tokenizer(prompt_string, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    prompt_len = inputs["input_ids"].shape[-1] # Length of the prompt to ignore later

    stop_criteria = StoppingCriteriaList([StopOnToken(tokenizer, "<end>")])

    for attempt in range(Retries):
        try:
            max_tokens = min(4096, dynamic_max_tokens * (2 ** attempt))

            gen_cfg = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    stopping_criteria=stop_criteria
                )
            
            # ***CRITICAL FIX 3: Only decode the generated part (after prompt_len)***
            generated_token_ids = outputs[0][prompt_len:]
            
            # ***CRITICAL FIX 2: skip_special_tokens=False to preserve <start>/<end> markers***
            raw = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            generated_text = raw.strip()

            # 1) Preferred extraction: explicit <start> ... <end>
            # Use [\s\S]*? to capture multiline content lazily, ensuring <end> is found
            match = re.search(r"<start>\s*([\s\S]*?)\s*<end>", generated_text, flags=re.S)
            
            if match:
                json_text = match.group(1).strip()
            else:
                # Fallback: The original output is used if <end> wasn't generated
                # But we strip <start> if it exists to clean the JSON content
                if generated_text.startswith("<start>"):
                    generated_text = generated_text[len("<start>"):].strip()

                m = re.search(r"(\{[\s\S]*\})", generated_text, flags=re.S)
                if m:
                    # Find the first balanced JSON object
                    json_text = m.group(1).strip()
                else:
                    raise ValueError("Could not find JSON object or <start>...</end> delimiters.")

            # Remove common trailing commas before } or ]
            json_text = re.sub(r',\s*([\]\}])', r'\1', json_text)

            return json.loads(json_text)

        except Exception as e:
            last_error = e
            debug_raw = raw.strip() if raw else "<no raw output>"
            print(f"[WARN] Attempt {attempt+1} failed: {type(e).__name__}: {e}\nRaw output:\n{debug_raw}\n")
    else:
        raise RuntimeError(f"Failed after {Retries} attempts.\nLast error: {last_error}\nRaw: {raw}")


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
