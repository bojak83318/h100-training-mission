#!/usr/bin/env python3
"""
GLM-4.7-Flash Training on H100 (80GB VRAM) - Pure PEFT/Transformers
No Unsloth, No bitsandbytes - Full bf16 training
Authorization: TPM-BMAD-2026-Q1-GO
"""

import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# ============================================
# H100 CONFIGURATION (80GB VRAM)
# ============================================
MODEL_NAME = "zai-org/GLM-4.7-Flash"
OUTPUT_DIR = "/workspace/outputs/glm4_rust_agent_h100"

# H100-optimized settings (no quantization needed)
MAX_SEQ_LENGTH = 8192   # Full context for complex tool calls
LORA_RANK = 16          # Higher rank for better quality
LORA_ALPHA = 32         # 2x rank
BATCH_SIZE = 2          # Conservative for large model
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2

# Data paths
RAW_DATA = "/workspace/data/rust_agent_native_format.jsonl"
TRAIN_DATA_DEFAULT = "/workspace/data/rust_agent_native_format_train.jsonl"
TRAIN_DATA_RW = "/workspace/outputs/rust_agent_native_format_train.jsonl"


def load_jsonl(path):
    """Load JSONL dataset"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    print("\n" + "=" * 60)
    print("GLM-4.7-Flash Training on H100 (Pure PEFT)")
    print("=" * 60)
    
    # 1. Resolve Data Path
    train_data_path = None
    if os.path.exists(TRAIN_DATA_DEFAULT):
        print(f"   ✅ Found pre-split training data: {TRAIN_DATA_DEFAULT}")
        train_data_path = TRAIN_DATA_DEFAULT
    elif os.path.exists(TRAIN_DATA_RW):
        print(f"   ✅ Using existing RW training split: {TRAIN_DATA_RW}")
        train_data_path = TRAIN_DATA_RW
    elif os.path.exists(RAW_DATA):
        print(f"   📊 Creating splits from {RAW_DATA}...")
        try:
            with open(RAW_DATA, 'r') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            train_lines = total_lines - 2000
            
            os.makedirs(os.path.dirname(TRAIN_DATA_RW), exist_ok=True)
            with open(TRAIN_DATA_RW, 'w') as f:
                f.writelines(lines[:train_lines])
            
            val_path = "/workspace/outputs/rust_validation_holdout.jsonl"
            with open(val_path, 'w') as f:
                f.writelines(lines[train_lines:])
                
            print(f"   ✅ Created: {train_lines} train samples, 2000 validation samples")
            train_data_path = TRAIN_DATA_RW
        except Exception as e:
            print(f"   ❌ Error creating splits: {e}")
            # Fallback to Raw Data if we can't split (unlikely to work well but as last resort)
            train_data_path = RAW_DATA
    else:
        print(f"   ❌ Error: Could not find training data at {RAW_DATA} or {TRAIN_DATA_DEFAULT}")
        exit(1)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mode: Full bf16 (no quantization)")
    print(f"  LoRA Rank: {LORA_RANK}")
    print(f"  Max Seq Length: {MAX_SEQ_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print("=" * 60)
    
    # Disable wandb for now
    os.environ["WANDB_MODE"] = "disabled"
    
    # Set HuggingFace cache to writable directory (OVHcloud runs as non-root)
    cache_dir = "/workspace/outputs/.cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    
    # Disable aggressive transfer to avoid stalls
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    hf_token = os.environ.get("HF_TOKEN")
    
    # Prefer local volume path if available
    LOCAL_MODEL_PATH = "/workspace/huggingface-models"
    actual_model_path = MODEL_NAME
    if os.path.isdir(LOCAL_MODEL_PATH):
        print(f"   ✅ Using local volume for weights: {LOCAL_MODEL_PATH}")
        # Ensure all shards are present (idempotent)
        from huggingface_hub import snapshot_download
        try:
            print(f"   📥 Syncing shards for {MODEL_NAME}...")
            snapshot_download(
                MODEL_NAME, 
                local_dir=LOCAL_MODEL_PATH, 
                token=hf_token,
                max_workers=8
            )
            actual_model_path = LOCAL_MODEL_PATH
        except Exception as e:
            print(f"   ⚠️ Could not sync shards locally: {e}. Falling back to default loader.")
    
    print(f"\n📥 Loading model from {actual_model_path}...")
    print(f"   Searching in cache: {os.environ.get('HF_HOME')}")
    
    # Use explicit device map and pass token explicitly
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}, 
        trust_remote_code=True,
        token=hf_token,
        # attn_implementation="flash_attention_2", # Use SDPA (default)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        actual_model_path,
        trust_remote_code=True,
        token=hf_token,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"   VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    print(f"   VRAM after LoRA: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n📊 Loading dataset...")
    train_data = load_jsonl(train_data_path)
    print(f"   Train samples: {len(train_data)}")
    
    train_dataset = Dataset.from_list(train_data)
    
    # Training arguments for H100
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=0.03,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,  # Full bf16 training
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="adamw_torch",  # Standard AdamW (no bitsandbytes)
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="none",  # Disable wandb
    )
    
    print("\n🚀 Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # Disable packing for stability
    )
    
    trainer.train()
    
    print("\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"   Final VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
