import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# Disable unnecessary warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset (larger subset for more extensive knowledge)
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train[:5000]")  # 5x more data

# TinyLlama-1.1B (no auth needed)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer with padding setup
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization for VRAM efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # float16 saves VRAM vs bfloat16
    bnb_4bit_use_double_quant=False,       # Disabled for lower memory
)

# Load model with optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# LoRA config (reduced size for 4GB VRAM)
peft_config = LoraConfig(
    r=8,                  # Reduced from 16
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Only target these to save memory
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ~0.5M params trainable

# Formatting for TinyLlama ChatML
def format_instruction(example):
    # Handle the actual dataset columns: system, user, assistant
    system_msg = example.get('system', 'You are a financial assistant.')
    user_msg = example['user']
    assistant_msg = example['assistant']
    
    text = f"<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n{assistant_msg}</s>"
    return {"text": text}

# Tokenize with truncation (shorter seq_len = less VRAM)
def tokenize(examples):
    return tokenizer(
        examples["text"],
        max_length=512,          # Reduced from 1024
        truncation=True,
        padding="max_length",
    )

# Apply formatting and tokenization
dataset = dataset.map(format_instruction)
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args optimized for RTX 3050
args = TrainingArguments(
    output_dir="./tinyllama-finance",
    per_device_train_batch_size=1,      # Critical for 4GB VRAM
    gradient_accumulation_steps=8,      # Compensate for small batch
    num_train_epochs=1,                 # Keep at 1 epoch with more data
    learning_rate=1e-4,                 # Lower rate for stability
    optim="paged_adamw_8bit",           # Memory-efficient optimizer
    logging_steps=10,
    save_strategy="steps",
    fp16=True,                          # float16 to save VRAM
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)

# Start training
print("Training... (Ctrl+C to stop early)")
trainer.train()

# Save adapter only (saves space)
model.save_pretrained("./tinyllama-finance-adapter-v2")  # Version 2 with more data