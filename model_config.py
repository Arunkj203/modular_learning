
# model_config.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM ,StoppingCriteria, StoppingCriteriaList 
import os , requests , re , json

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

#BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # Original Model

BASE_MODEL = "meta-llama/CodeLlama-34b-hf" # Larger Model

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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,token=HUGGINGFACEHUB_API_TOKEN)

    print(f"Loading model {BASE_MODEL} on {DEVICE} (FP16)...")
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

class StopOnToken(StoppingCriteria):
    def __init__(self, tokenizer, stop_token):
        self.tokenizer = tokenizer
        self.stop_token = stop_token
        self.stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
        
    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last tokens match our stop token
        if input_ids.shape[1] < len(self.stop_token_ids):
            return False
            
        # Get the recent tokens that could match our stop token
        recent_tokens = input_ids[0, -len(self.stop_token_ids):].tolist()
        return recent_tokens == self.stop_token_ids


def generate_text(model ,tokenizer, system_prompt, user_prompt,dynamic_max_tokens=200):

    prompt = f"""SYSTEM:
        {system_prompt}

        USER:
        {user_prompt}


        RESPONSE:
        """
    last_error = None
    raw = None
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    for attempt in range(Retries):

        try:
        
            max_tokens = min(4096, dynamic_max_tokens * (2 ** attempt))
        
            gen_cfg = GenerationConfig(
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    stopping_criteria=StoppingCriteriaList([
                StopOnToken(tokenizer, "<end>")
            ])
                
                )

            # Decode to string
            raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # print(raw)

            # Extract text after RESPONSE:
            generated_text = raw.split("RESPONSE:")[-1].strip()

            """
            Extract JSON array of primitives from raw LLM output wrapped with <start> and <end>
            """
            # Extract text between <start> and <end>
            match = re.search(r"<start>(.*?)<end>", generated_text, flags=re.S)
            if not match:
                raise ValueError("Could not find <start> ... <end> in raw output")

            json_text = match.group(1).strip()

            # Remove trailing commas before } or ]
            json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)

            return json.loads(json_text)

        except Exception as e:
                    last_error = e
                    print(f"[WARN] Attempt {attempt+1} failed: {e}\nRaw:{raw.strip()}\n")
    else:
        raise RuntimeError(f"Failed\nError:{last_error}.\nRaw Output:\n{raw.strip()}")


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
