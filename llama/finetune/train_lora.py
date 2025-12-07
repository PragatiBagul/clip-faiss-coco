# llama/finetune/train_lora.py
import os
import argparse
import logging
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, required=False)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--use_8bit", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    return p.parse_args()

def load_jsonl_as_dataset(path):
    ds = load_dataset("json", data_files={"train": path})
    return ds["train"]

def prepare_examples(example, tokenizer, max_seq_len):
    """
    Given an example with fields: context (string), target (string) -> produce a single input string
    We'll create the prompt as: <context>\n\nAnswer: <target>
    """
    prompt = f"{example['context']}\n\nAnswer:"
    full = prompt + " " + example["target"]
    tokenized = tokenizer(full, truncation=True, max_length=max_seq_len)
    # we will train on full sequence with labels same as input_ids (causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer and model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model (optionally prepare for k-bit training)
    if args.use_8bit:
        # for 8-bit, accelerate + bitsandbytes needed. We'll use prepare_model_for_kbit_training
        model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto")

    # Apply LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj","k_proj","o_proj","dense","mlp","dense_h_to_4h"],  # general targets, adapt to model
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)
    logger.info("PEFT LoRA model prepared.")

    # Load dataset(s)
    print("Loading datasets...")
    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    val_ds = None
    if args.validation_file:
        val_ds = load_dataset("json", data_files=args.validation_file)["train"]

    # Tokenize dataset
    def tok_map(ex):
        return prepare_examples(ex, tokenizer, args.max_seq_len)

    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names, batched=False)
    if val_ds is not None:
        val_tok = val_ds.map(tok_map, remove_columns=val_ds.column_names, batched=False)
    else:
        val_tok = None

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    # TrainingArguments compatible with accelerate / Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=not args.use_8bit and torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_tok is not None else "no",
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save only the adapter (LoRA weights)
    peft_output = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(peft_output)
    logger.info(f"Saved LoRA adapter to {peft_output}")

if __name__ == "__main__":
    main()
