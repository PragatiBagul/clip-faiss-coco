# LLaMA / LoRA module — clip-faiss-coco (v2)

This folder contains the LLaMA integration used to generate captions and explanations on top of the CLIP+FAISS retrieval system.

## What’s here (expected)
- `prompts/` — prompt templates used for generation (caption / explanation / summarization).
- `inference/` — LLaMA inference wrapper plus prompt-builder utilities.
- `finetune/` — LoRA/PEFT training scripts and dataset loaders.
- `data/train.jsonl` — sample training data (small JSONL).
- `configs/` — model & peft configuration files.
- `experiments/` — (optional) runs, adapters, and sample outputs.

## Model choices & licensing
- **Base model:** Use a model you are licensed to use. Examples:
  - Meta Llama-2 (via Hugging Face) — read Meta license terms for usage restrictions.
  - Vicuna / Mistral / other permissive HF-hosted models.
- **Important:** If using LLaMA family weights, ensure you have the proper license/permissions. If unsure, prefer Llama-2 HF repos (review their license).

## Quickstart — zero-shot inference
1. Install dependencies:
```bash
pip install -r requirements.txt
# ensure transformers, torch, peft, datasets, accelerate (optional) are installed
