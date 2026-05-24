"""
train.py

Finetuning script for CivilTime-Benchmark models.
Uses Hugging Face TRL SFTTrainer + PEFT (LoRA).
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from utils import load_jsonl_dataset, format_instruction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", default="microsoft/phi-4")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--token", help="HF Token", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    if args.token:
        from huggingface_hub import login
        login(args.token)

    # Load Data
    print(f"Loading data from {args.data_path}")
    ds = load_jsonl_dataset(args.data_path)
    ds = ds.map(format_instruction, num_proc=4, remove_columns=ds.column_names)
    print(f"Sample: {ds[0]['text'][:200]}...")

    # Load Model
    print(f"Loading model {args.model_name}")
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.use_4bit else None
    )
    model.config.use_cache = False

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Adjust for specific models if needed
    )

    # Config
    # SFTConfig replaces TrainingArguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        packing=False,
        dataset_text_field="text", # Needed for SFTTrainer
        max_seq_length=1024        # Needed for SFTTrainer
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
