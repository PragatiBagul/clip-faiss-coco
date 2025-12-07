#!/usr/bin/env python3
"""
Train a LoRA adapter to teach the LLM to use retrieved captions as context.

Key features:
- Uses `accelerate` for distributed / mixed-precision training.
- Uses PEFT (LoraConfig + get_peft_model).
- Expects train/val files in JSONL format (see llama/data/make_lora_dataset.py).
- Saves adapter only to `output_dir/run_<timestamp>/adapter/`.
"""

import os
import time
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import evaluate

# Simple dataset
class LoraJsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512, prompt_template=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or ("Context:\n{context}\n\n"
                                                   "Write a structured JSON with keys: ShortCaption, DetailedCaption, Tags.\n\n"
                                                   "Ground truth:\n{expanded}")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                prompt = self.prompt_template.format(context=obj.get("context",""), expanded=obj.get("expanded",""))
                # For causal LM training we combine prompt + target; label tokens for the target portion only
                input_ids = tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze(0)
                # In this implementation we train to predict the whole target (prompt contains both)
                # For fine control you can separate prompt/labels with masks; here simple next-token modeling is used
                self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    # pad to max len in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    # labels are same as input_ids for causal LM (teacher forcing)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, default="llama/data/train.jsonl")
    p.add_argument("--validation_file", type=str, default="llama/data/val.jsonl")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b")
    p.add_argument("--output_dir", type=str, default="llama/experiments/runs")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--use_8bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_every_n_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else None)
    device = accelerator.device

    #Tokenizer & dataset (tokenizer must be shared with model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading datasets...")
    train_ds = LoraJsonlDataset(args.train_file, tokenizer, max_length=args.max_length)
    val_ds = LoraJsonlDataset(args.validation_file, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    print("Loading base model...")
    if args.use_8bit and torch.cuda.is_available():
        # prepare for 8-bit training
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True)
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True)

    # PEFT/LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "dense"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
    if args.max_steps > 0:
        total_steps = args.max_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=total_steps)

    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    # Training loop
    global_step = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = Path(args.output_dir) / f"run_{run_id}"
    out_adapter = out_run / "adapter"
    out_run.mkdir(parents=True, exist_ok=True)

    metrics = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    print("Start training...")
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")

            # Optional save
            if args.save_every_n_steps > 0 and (global_step % args.save_every_n_steps == 0):
                # Save adapter only
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(out_adapter)
                print(f"Saved adapter at step {global_step} -> {out_adapter}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        # End epoch: evaluate on val set (sample small subset for speed)
        print(f"Evaluating after epoch {epoch} ...")
        model.eval()
        references = []
        predictions = []
        with torch.no_grad():
            for vb in val_loader:
                # generate using model.generate; convert to CPU for tokenizer decode
                # Note: some setups require moving to device; accelerate handles it
                input_ids = vb["input_ids"]
                attention_mask = vb["attention_mask"]
                # generate 1 candidate
                gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=False)
                texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                # For simplicity we treat whole generated text as prediction; references we use labels decode
                preds = texts
                refs = tokenizer.batch_decode(vb["labels"], skip_special_tokens=True)
                predictions.extend(preds)
                references.extend(refs)

        # compute BLEU/ROUGE on this epoch
        try:
            bleu_score = metrics.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
        except Exception:
            bleu_score = None
        try:
            rouge_result = rouge.compute(predictions=predictions, references=references)
        except Exception:
            rouge_result = {}

        print(f"Epoch {epoch} eval BLEU={bleu_score}, ROUGE={rouge_result}")

        model.train()

    # final save of adapter
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(out_adapter)
    print(f"Training complete. Adapter saved to {out_adapter}")
