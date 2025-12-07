# llama/finetune/eval_lora.py
import os
import argparse
import csv
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from peft import PeftModel
from transformers import AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True)
    p.add_argument("--val_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    dataset = load_dataset("json", data_files=args.val_file)["train"]

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    out_rows = []
    for ex in dataset:
        context = ex["context"]
        target = ex.get("target") or ex.get("image_caption") or ex.get("expanded", "")
        prompt = context + "\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        # strip prompt naive
        if pred.startswith(prompt):
            pred = pred[len(prompt):].strip()
        out_rows.append({"context": context, "target": target, "prediction": pred})

    # compute metrics
    refs = [r["target"] for r in out_rows]
    hyps = [r["prediction"] for r in out_rows]
    bleu_res = bleu.compute(predictions=hyps, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=hyps, references=refs)
    bert_res = bertscore.compute(predictions=hyps, references=refs, lang="en")

    # save eval.csv
    csv_path = os.path.join(args.output_dir, "eval.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["context", "target", "prediction"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    # save metrics summary
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"bleu": bleu_res, "rouge": rouge_res, "bertscore": bert_res}, fh, indent=2)

    print("Saved eval.csv and metrics.json in", args.output_dir)

if __name__ == "__main__":
    main()
