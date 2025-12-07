# place under llama/finetune/
import argparse
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    _has_peft = True
except Exception:
    _has_peft = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()
    if not _has_peft:
        raise RuntimeError("peft package required. Install via `pip install peft`")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model in 8-bit if available to save memory
    model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build dataset (simple jsonl loader)
    train_ds = load_dataset("json", data_files=args.train_file)["train"]
    def preprocess(ex):
        prompt = f"Context:\n{ex['context']}\n\nInstruction: Generate a JSON object with Title, ShortCaption, DetailedCaption, Tags.\nOutput:\n"
        full = prompt + ex.get("caption", "")
        tok = tokenizer(full, truncation=True, max_length=args.max_seq_length, padding="max_length")
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, batched=False)
    # Data collator and Trainer/accelerate loop — keep simple: use Trainer from transformers for clarity
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    trainer.train()
    # Save LoRA adapter only
    model.save_pretrained(args.output_dir)
    logger.info(f"Saved adapter to {args.output_dir}")

if __name__ == "__main__":
    main()
