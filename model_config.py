
# model_config.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM ,StoppingCriteria, StoppingCriteriaList 
import os , requests 

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


def generate_text(model ,tokenizer, system_prompt, user_prompt,max_tokens=200):

    prompt = (
    f"<s>[SYSTEM]\n{system_prompt}\n"
    f"[USER]\n{user_prompt}\n"
    f"[ASSISTANT]\n"
    "Output starts now. Do not repeat the problem. Only produce the JSON output enclosed in <start> and <end>.\n"

    )


    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
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
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	
    #print(raw_output)

    # Extract text after RESPONSE:
    generated_text = raw_output.split("[ASSISTANT]")[-1].strip()

    return generated_text  # Can now pass to json.loads(json_text)

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
