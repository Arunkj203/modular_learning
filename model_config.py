
# model_config.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import os , requests , re




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
judge_model_Name = "gpt-4-jurassic"

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

    judge_model = AutoModelForCausalLM.from_pretrained(judge_model_Name, device_map="auto")


    return model, judge_model , tokenizer

# -----------------------------
# Helper function for generation
# -----------------------------
def generate_text(model ,tokenizer, system_prompt, user_prompt,max_tokens=200):

    prompt = f"""SYSTEM:
        {system_prompt}

        USER:
        {user_prompt}

        RESPONSE:
        """


    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id

        )

    # Decode to string
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
	
   # print(raw_output)
    # Extract text after RESPONSE:
    after_response = raw_output.split("RESPONSE:")[-1]

    

    # Extract text between <start> and <end>
    # match = re.search(r'<start>(.*?)<end>', after_response, flags=re.S)
    #if not match:
    #    raise ValueError(f"No <start> ... <end> JSON block found in output==>\n{after_response}.")

    #json_text = match.group(1).strip()

    # Remove trailing commas before } or ]
    #json_text = re.sub(r',(\s*[\}\]])', r'\1', json_text)

    return after_response  # Can now pass to json.loads(json_text)

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
