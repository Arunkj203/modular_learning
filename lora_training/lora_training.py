import os
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig 

from .preprocess_json import preprocess_phase1_dataset,preprocess_phase2_dataset,preprocess_phase3_dataset
from ..model_config import get_model_and_tokenizer


def train_lora_adapter(model, tokenizer, data, adapter_name,
                       output_root="Adapters", lr=2e-4, epochs=3, batch_size=4, max_length=256):
    """
    Generic LoRA training block — used by all phases.
    Args:
        model: Base model (torch model)
        tokenizer: Tokenizer for the model
        train_data: list[dict] or HuggingFace Dataset
        val_data: list[dict] or HuggingFace Dataset
        adapter_name: name for saving adapter (e.g. "phase1_svamp")
        output_root: base directory to save adapter folders
        lr, epochs, batch_size, max_length: training hyperparameters
    Returns:
        trained_model: Model with trained LoRA adapter loaded
    """

   
    train_data = data["train"]
    val_data = data["val"]
    
    # Prepare output directory
    output_dir = os.path.join(output_root, adapter_name)
    os.makedirs(output_dir, exist_ok=True)

    # LoRA adapter config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config
    sft_config = SFTConfig(
        max_length=max_length,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        dataset_text_field="text",
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        args=sft_config,
        peft_config=peft_config,
    )

    print(f"[LoRA Training] Starting adapter: {adapter_name}")
    print(f"  -> Train size: {len(train_data)}, Val size: {len(val_data)}")

    trainer.train()
    trainer.model.save_pretrained(output_dir)

    print(f"[LoRA Training] Adapter '{adapter_name}' saved at {output_dir}")

    # Cleanup: free GPU cache (optional)
    torch.cuda.empty_cache()


def phase_train(files,phase,adapter_name):
    
    print(f"[Phase1] Loading {len(files)} files...")

    

    dataset = None
    if phase == 1:
        dataset = preprocess_phase1_dataset(files)
    elif phase == 2:
        dataset = preprocess_phase2_dataset(files)

    elif phase == 3:
        dataset = preprocess_phase3_dataset(files)
    else:
        raise ValueError("Phase must be 1, 2, or 3") 

    # ---- Load Base Model ----
    model, tokenizer = get_model_and_tokenizer()
    model.to("cuda")

    # ---- Train LoRA ----
    train_lora_adapter(
        model=model,
        tokenizer=tokenizer,
        data=dataset,
        adapter_name=adapter_name
    )
